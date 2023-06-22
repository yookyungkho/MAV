import numpy as np
import pandas as pd
import random
from tqdm import tqdm 
import torch
import types 
from torch.utils import data as t_data
import os
from omegaconf import OmegaConf
import argparse
import warnings

import nlpaug.augmenter.word as naw

import transformers
from transformers import AutoTokenizer

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    transformers.set_seed(random_seed)

class RandomAugMentation():
    def __init__(self, augmentation_args) -> None:
        """
        Warning : You cannot use same augmentation as weak_aug and strong_aug both 
        Example : 
        data_args = {
            "strong_aug" : "mask*dropout",
            "strong_aug_init_mag" : 0.1
        }   
        """
        self.augmentation_args = augmentation_args
        self.aug_func_dict = self.__init_aug_func_dict()
        self.aug_func_list = [self.aug_func_dict[aug_name] for aug_name in self.augmentation_args.strong_aug.split("*")]
        [self.update_aug_magnitude(aug_func, self.augmentation_args.strong_aug_mag, aug_name) for aug_func, aug_name in zip(self.aug_func_list, self.augmentation_args.strong_aug.split("*"))]

        print(">>> Strong Augmentation : ", augmentation_args.strong_aug)
        print(">>> Strong Augmentation Magnitude : ", augmentation_args.strong_aug_mag)

    def __init_aug_func_dict(self) :
        # sourcery skip: inline-immediately-returned-variable
        ## Word Level
        self.random_word_swap = naw.RandomWordAug(action='swap')
        self.random_word_delete = naw.RandomWordAug(action='delete')
        self.bertaug = naw.ContextualWordEmbsAug(model_path='roberta-base', device='cuda', action="substitute", batch_size=32)
        self.backtranslation = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en', device='cuda', batch_size=32, max_length=256)
        self.backtranslation.model.translate_one_step_batched = types.MethodType(translate_one_step_batched, self.backtranslation.model)
        
        ## Tokenized Level
        class identity_aug() :
            def augment(self, text) :
                return [text]

        self.mask = identity_aug()
        self.dropout = identity_aug()

        aug_func_dict = {
            "wordswap"          : self.random_word_swap,
            "worddelete"        : self.random_word_delete,
            "wordsubstitute"    : self.synonym_word_substitute,
            "bertaug"           : self.bertaug,
            "backtranslation"   : self.backtranslation,
        }
        return aug_func_dict

    def update_aug_magnitude(self, aug_func , aug_mag : float, aug_name : str):
        """
        example 
        ------------
        aug_func : self.aug_func_dict["synonym"]
        aug_mag : 0.2
        aug_name : "synonym"
        return : naw.RandomWordAug(action='swap', aug_p=0.2)
        """
        assert aug_name in self.aug_func_dict.keys(), f"aug_name should be in {self.aug_func_dict.keys()}"

        if aug_name in {"wordswap", "worddelete", "wordsubstitute", "bertaug"} :
            aug_func.aug_p = aug_mag
        elif aug_name in {"backtranslation"} :
            pass
        else : 
            raise NotImplementedError
 
    def augment(self, sentences) :
        aug_func = np.random.choice(self.aug_func_list)
        return aug_func.augment(sentences)

def translate_one_step_batched(self, data, tokenizer, model) : # methods for replacing the original translate_one_step_batched method in backtranslation. transformers version issue
    tokenized_texts = tokenizer(data, padding=True, truncation=True, return_tensors='pt')
    tokenized_dataset = t_data.TensorDataset(*(tokenized_texts.values()))        
    tokenized_dataloader = t_data.DataLoader(
        tokenized_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=1
    )

    all_translated_ids = []
    with torch.no_grad():
        for batch in tokenized_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids = batch[0]
            attention_mask = batch[2]
            translated_ids_batch = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_length
            )

            all_translated_ids.append(
                translated_ids_batch.detach().cpu().numpy()
            )

    all_translated_texts = []
    for translated_ids_batch in all_translated_ids:
        translated_texts = tokenizer.batch_decode(
            translated_ids_batch,
            skip_special_tokens=True
        )
        all_translated_texts.extend(translated_texts)

    return all_translated_texts


def tokenize_multipart_input(
    input_text_list, 
    max_length, 
    tokenizer, 
    template=None,
    label_word_list=None, 
):
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    input_ids = []
    attention_mask = []
    token_type_ids = [] # Only for BERT
    mask_pos = None # Position of the mask token

    """
    Concatenate all sentences and prompts based on the provided template.
    Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
    *xx* represent variables:
        *cls*: cls_token
        *mask*: mask_token
        *sep*: sep_token
        *sep+*: sep_token, also means +1 for segment id
        *sent_i*: sentence i (input_text_list[i])
        *sent-_i*: same as above, but delete the last token
        *sentl_i*: same as above, but use lower case for the first word
        *sentl-_i*: same as above, but use lower case for the first word and delete the last token
        *+sent_i*: same as above, but add a space before the sentence
        *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
        *label_i*: label_word_list[i]
        *label_x*: label depends on the example id (support_labels needed). this is only used in GPT-3's in-context learning

    Use "_" to replace space.
    PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
    """
    assert template is not None

    special_token_mapping = {
        'cls': tokenizer.cls_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id, 'sep+': tokenizer.sep_token_id, 
    }
    template_list = template.split('*') # Get variable list in the template
    segment_id = 0 # Current segment id. Segment id +1 if encountering sep+.

    for part_id, part in enumerate(template_list):
        new_tokens = []
        segment_plus_1_flag = False
        if part in special_token_mapping:
            if part == 'cls' and 'T5' in type(tokenizer).__name__:
                # T5 does not have cls token
                continue
            new_tokens.append(special_token_mapping[part])
            if part == 'sep+':
                segment_plus_1_flag = True
        elif part[:6] == 'label_':
            # Note that label_word_list already has extra space, so do not add more space ahead of it.
            label_id = int(part.split('_')[1])
            label_word = label_word_list[label_id]
            new_tokens.append(label_word)
        elif part[:7] == 'labelx_':
            instance_id = int(part.split('_')[1])
            label_id = support_labels[instance_id]
            label_word = label_word_list[label_id]
            new_tokens.append(label_word)
        elif part[:5] == 'sent_':
            sent_id = int(part.split('_')[1])
            new_tokens += enc(input_text_list[sent_id]) 
        elif part[:6] == '+sent_':
            # Add space
            sent_id = int(part.split('_')[1])
            new_tokens += enc(' ' + input_text_list[sent_id])
        elif part[:6] == 'sent-_':
            # Delete the last token
            sent_id = int(part.split('_')[1])
            new_tokens += enc(input_text_list[sent_id][:-1])
        elif part[:6] == 'sentl_':
            # Lower case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(text)
        elif part[:7] == '+sentl_':
            # Lower case the first token and add space 
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(' ' + text)
        elif part[:7] == 'sentl-_':
            # Lower case the first token and discard the last token
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(text[:-1])
        elif part[:6] == 'sentu_':
            # Upper case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].upper() + text[1:]
            new_tokens += enc(text)
        elif part[:7] == '+sentu_':
            # Upper case the first token and add space
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].upper() + text[1:]
            new_tokens += enc(' ' + text)
        elif part == ":" :
            new_tokens += enc(part)
        elif part == "" :
            pass
        else : 
            # Other cases
            raise ValueError('Unknown template part: ' + part)

        if part[:4] == 'sent' or part[1:5] == 'sent':
            # If this part is the sentence, limit the sentence length
            sent_id = int(part.split('_')[1])

        input_ids += new_tokens
        attention_mask += [1 for i in range(len(new_tokens))]
        token_type_ids += [segment_id for i in range(len(new_tokens))]

        if segment_plus_1_flag:
            segment_id += 1

    while len(input_ids) < max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        token_type_ids.append(0)

    # Truncate
    if len(input_ids) > max_length:
        print(">>> Some examples are too long. Truncate them.")
        # Default is to truncate the tail
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        token_type_ids = token_type_ids[:max_length]

    # Find mask token
    try : 
        mask_pos = [input_ids.index(tokenizer.mask_token_id)]
        # Make sure that the masked position is inside the max_length
        assert mask_pos[0] < max_length
    except :
        mask_pos = None
        warnings.warn("No mask token in the template. This may not be intended.")

    result = {'input_ids': input_ids, 'attention_mask': attention_mask}
    if 'BERT' in type(tokenizer).__name__:
        # Only provide token type ids for BERT
        result['token_type_ids'] = token_type_ids

    result['mask_pos'] = mask_pos       

    return result["input_ids"]


def load_data(seed_data_dir) :
    data = pd.read_csv(os.path.join(seed_data_dir, "unlabeled.csv"), header = None)
    if data.shape[1] == 2 :
        return data.iloc[:, 1].to_list()
    else : 
        data_first = data.iloc[:, 1].to_list()
        data_second = data.iloc[:, 2].to_list()
        return list(zip(data_first, data_second))

def augment_data(data_args, data) :
    print(">>> The length of the original data is {}".format(len(data)))
    print(">>> Example : {}".format(data[0]))
    augmenter = RandomAugMentation(data_args)



    if sum([aug_name in data_args.strong_aug_list for aug_name in {"wordswap", "worddelete", "wordsubstitute"}]) == 1 : # if naive augmentation applied
        if not isinstance(data[0], str) : # if data is list of sentences
            data_first = [sent[0] for sent in data]
            data_second = [sent[1] for sent in data]
            result_first = [augmenter.augment(sent) for sent in tqdm(data_first)]
            result_second = [augmenter.augment(sent) for sent in tqdm(data_second)]
            result = [[first[0], second[0]] for first, second in zip(result_first, result_second)]
        else :
            result = [augmenter.augment(sent) for sent in data]
        print(">>> The length of the augmented data is {}".format(len(result)))
        print(">>> Example : {}".format(result[0]))
        return result

    else : # if model-based augmentation is applied for batch
        if not isinstance(data[0], str) : # if data is list of sentences
            data_first = [sent[0] for sent in data]
            data_second = [sent[1] for sent in data]
            result_first = augmenter.augment(data_first)
            result_second = augmenter.augment(data_second)
            result = [[first, second] for first, second in zip(result_first, result_second)]
        else :
            result = augmenter.augment(data)
            print(f">>> The length of the augmented data is {len(result)}") 
            print(f">>> Example : {result[0]}")
            result = [[text] for text in result]
        return result
    

def tokenize_data(tokenize_args, data) :
    tokenizer = AutoTokenizer.from_pretrained(tokenize_args.tokenizer_name)
    return [tokenize_multipart_input(
        input_text_list = sent, 
        tokenizer       = tokenizer, 
        max_length      = tokenize_args.max_length,
        template        = tokenize_args.template,
            ) for sent in data]

def save_data(augmentation_name, save_dir, tokenized_data, dir_args) :
    if dir_args.additinal_save_name :
        save_dir = os.path.join(save_dir, f"unlabeled_{augmentation_name}_{dir_args.additinal_save_name}")
    else : 
        save_dir = os.path.join(save_dir, f"unlabeled_{augmentation_name}")
    np.save(save_dir, tokenized_data)
    return save_dir

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, default='augmentation_goemotions.yaml')
    args, _ = parser.parse_known_args()
    total_args = OmegaConf.load(f'./{args.config_dir}')
    data_dir_list = os.listdir(os.path.join(total_args.dir_args.data_dir, total_args.dir_args.data_name))
    for seed in [13, 21, 42, 87, 100] : 
        print(f"\n >>>>>>>> Start Augmentation & Tokenization for seed : {seed} ")
        for strong_aug_name in total_args.augmentation_args.strong_aug_list :
            total_args.augmentation_args.strong_aug = strong_aug_name
            total_args.augmentation_args.seed = seed
            set_seed(total_args.augmentation_args.seed)
            seed_data_dir = [data_dir for data_dir in data_dir_list if str(seed) in data_dir][0]
            seed_data_dir = os.path.join(total_args.dir_args.data_dir, total_args.dir_args.data_name, seed_data_dir)
            data = load_data(seed_data_dir)
            augmented_data = augment_data(total_args.augmentation_args, data)
            tokenized_data = tokenize_data(total_args.tokenize_args, augmented_data)
            save_dir = save_data(strong_aug_name, seed_data_dir, tokenized_data, total_args.dir_args)
            print(f">>> Augmentation & Tokenization done : {save_dir}")
if __name__ == "__main__" :
    main()