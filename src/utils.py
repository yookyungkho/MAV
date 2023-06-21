import json
import torch
from transformers import PreTrainedTokenizer
from typing import Tuple
import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set_palette('muted')


# ---------------- Basic Functions ----------------
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def load_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data

def save_json(file_path, save_file):
    json.dump(save_file, open(file_path,'w'), indent=4)


# ---------------- Functions for input preprocessing ----------------
def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, mask_probability=0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )
    inputs = inputs.cpu()
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mask_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs.cuda(), labels.cuda()

def cut_aug_input(model_type, tokens, embeds, aug_tokens, aug_embeds):
    if 'roberta' in model_type:
        attention_mask = (tokens != 1).float()
        attention_mask2 = (aug_tokens != 1).float()
    else:
        attention_mask = (tokens > 0).float()
        attention_mask2 = (aug_tokens > 0).float()

    max_len_orig = int(torch.max(attention_mask.sum(dim=1)))
    max_len_aug = int(torch.max(attention_mask2.sum(dim=1)))

    max_len = max(max_len_orig, max_len_aug)

    return tokens[:, :max_len], embeds[:, :max_len, :], aug_tokens[:, :max_len], aug_embeds[:, :max_len, :]


# ---------------- Functions for output postprocessing ----------------
def mean_pooling(input_ids_weak, input_ids_strong, sent_weak, sent_strong):
    batch_size = input_ids_weak.size(0)
    num_mask = (input_ids_weak != 1).float().sum(dim=1).long() #except [PAD]
    num_mask_strong = (input_ids_strong != 1).float().sum(dim=1).long() # tensor([128, 128, 128, 119, 128, 128, 128, 128], device='cuda:0')
    mean_sent_weak, mean_sent_strong = [], []

    for b in range(batch_size):
        mean_sent_weak_b = sent_weak[b, :num_mask[b], :].mean(dim=0).unsqueeze(0)  # (1, dim)
        mean_sent_strong_b = sent_strong[b, :num_mask_strong[b], :].mean(dim=0).unsqueeze(0)  # (1, dim)

        mean_sent_weak.append(mean_sent_weak_b)
        mean_sent_strong.append(mean_sent_strong_b)

    mean_sent_weak = torch.cat(mean_sent_weak, dim=0)  # (batch_size, dim)
    mean_sent_strong = torch.cat(mean_sent_strong, dim=0)  # (batch_size, dim)

    return mean_sent_weak, mean_sent_strong


# ---------------- Functions for visualization ----------------
def draw_tsne_plot(tsne_np, y, label2word, tsne_dir, save_title, mode):
    save_dir = f"{tsne_dir}/{save_title}.{mode}"

    tsne_df = pd.DataFrame(tsne_np, columns = ['T1', 'T2'])
    tsne_df['y'] = y

    n_class = len(label2word)

    for i in range(n_class):
        globals()[f"tsne_df_{i}"] = tsne_df[tsne_df['y'] == i]

    palette = np.array(sns.color_palette("hls", n_class))
    plt.figure()

    for i in range(n_class):
        plt.scatter(globals()[f"tsne_df_{i}"]['T1'], globals()[f"tsne_df_{i}"]['T2'], color = palette[i], label = label2word[i])

    plt.xlabel('T1')
    plt.ylabel('T2')
    plt.savefig(save_dir, dpi=300)
    plt.close()