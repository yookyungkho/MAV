import numpy as np
import pandas as pd
import random
import torch
import os
from omegaconf import OmegaConf
import argparse

import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.model.word_stats as nmw

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
        self.synonym_word_substitute = naw.SynonymAug()

        ## Static Word Embedding Level
        # self.embed_word_substitute = naw.WordEmbsAug(model_type='word2vec', model_path=self.augmentation_args.word_embedding_dir, action="substitute", top_k=10)
        # self.embed_word_insert = naw.WordEmbsAug(model_type='word2vec', model_path=self.augmentation_args.word_embedding_dir, action="insert", top_k=10)

        # ## TFIDF Word Level
        # self.tfidf_word_substitute = naw.TfIdfAug(action="substitute", model_path=self.augmentation_args.data_dir, tokenizer=self.data_args.tokenizer.tokenize, reverse_tokenizer=self.data_args.tokenizer.convert_tokens_to_string, top_k=10)
        # self.tfidf_word_insert = naw.TfIdfAug(action="insert", model_path=self.augmentation_args.data_dir, tokenizer=self.data_args.tokenizer.tokenize, reverse_tokenizer=self.data_args.tokenizer.convert_tokens_to_string, top_k=10)

        # ## Text Level
        # self.ocr = nac.OcrAug(aug_char_p = 0.2)
        # self.keyboard = nac.KeyboardAug(aug_char_p = 0.2)
        # self.random_char_insert = nac.RandomCharAug(action="insert", aug_char_p = 0.2)
        # self.random_char_substitute = nac.RandomCharAug(action="substitute", aug_char_p = 0.2)
        # self.random_char_swap = nac.RandomCharAug(action="swap", aug_char_p = 0.2)
        # self.random_char_delete = nac.RandomCharAug(action="delete", aug_char_p = 0.2)
        
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
            # "embedsubstitute"   : self.embed_word_substitute,
            # "embedinsert"       : self.embed_word_insert,
            # "tfidfwordsubstitute"   : self.tfidf_word_substitute,
            # "tfidfwordinsert"   : self.tfidf_word_insert,
            # "ocr"               : self.ocr,
            # "keyboard"          : self.keyboard,
            # "charinsert"        : self.random_char_insert,
            # "charsubstitute"    : self.random_char_substitute,
            # "charswap"          : self.random_char_swap,
            # "chardelete"        : self.random_char_delete,
            # "mask"              : self.mask,
            # "dropout"           : self.dropout
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

        if aug_name in {"wordswap", "worddelete", "wordsubstitute", "embedsubstitute", "embedinsert", "tfidfwordsubstitute", "tfidfwordinsert"}:
            aug_func.aug_p = aug_mag
        elif aug_name in {"ocr", "keyboard", "charinsert", "charsubstitute", "charswap", "chardelete"}:
            aug_func.aug_char_p = aug_mag

    def augment(self, text) :
        aug_func = np.random.choice(self.aug_func_list)
        return aug_func.augment(text)


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
    mask_pos = [input_ids.index(tokenizer.mask_token_id)]
    # Make sure that the masked position is inside the max_length
    assert mask_pos[0] < max_length

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
    augmenter = RandomAugMentation(data_args)
    return [augmenter.augment(text) for text in data]

def tokenize_data(tokenize_args, data) :
    tokenizer = AutoTokenizer.from_pretrained(tokenize_args.tokenizer_name)
    return [tokenize_multipart_input(
        input_text_list = sent, 
        tokenizer       = tokenizer, 
        max_length      = tokenize_args.max_length,
        template        = tokenize_args.template,
            ) for sent in data]

def save_data(augmentation_name, save_dir, tokenized_data) :
    np.save(os.path.join(save_dir, f"unlabeled_{augmentation_name}"), tokenized_data)

def main() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, default='augmentation_trec')
    args, _ = parser.parse_known_args()
    total_args = OmegaConf.load(f'./{args.config_dir}.yaml')
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
            save_data(strong_aug_name, seed_data_dir, tokenized_data)
            print(f">>> Augmentation & Tokenization done : {seed}/{strong_aug_name}")
if __name__ == "__main__" :
    main()