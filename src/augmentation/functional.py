# Code is mainly adopted from https://github.com/moskomule/dda/tree/fasteraa/faster_autoaugment
# and https://openreview.net/forum?id=Ucx3DQbC9GH
""" `functional` contains deterministic functions
img image tensor `img` is expected to be CxHxW or BxCxHxW and its range should be [0, 1]
`mag=0` expects no transformation
"""

import functools
from typing import Optional

import numpy as np
from torch import nn
from torch.nn import functional as F
import torch.distributions.categorical as categorical

from ..utils import mask_tokens

# Note: if magnitude should be updated => ste(output, mag). If not, ste(output, input)

__all__ = ['eda_word_del', 'eda_word_swap', 'eda_word_del_swap', 'cutoff', 'cbert', 'backtrans', 'r3f', 'random_mask']

from typing import Tuple

import torch
from torch.autograd import Function


class _STE(Function):
    """
    StraightThrough Estimator
    
    * reference link: https://hongl.tistory.com/206
    """
    @staticmethod
    def forward(ctx,
                input_forward: torch.Tensor,
                input_backward: torch.Tensor) -> torch.Tensor:
        ctx.shape = input_backward.shape
        return input_forward

    @staticmethod
    def backward(ctx,
                grad_in: torch.Tensor) -> Tuple[None, torch.Tensor]:
        return None, grad_in.sum_to_size(ctx.shape)


def ste(input_forward: torch.Tensor,
        input_backward: torch.Tensor) -> torch.Tensor:
    """
    Straight-through estimator
    (for updating magnitude)
    :param input_forward:
    :param input_backward:
    :return:
    """

    return _STE.apply(input_forward, input_backward).clone()

def generate_noise(embed, epsilon=1e-5):
    noise = embed.data.new(embed.size()).normal_(0, 1) * epsilon
    noise.detach()
    noise.requires_grad_()
    return noise


# --------------------- Discrete Aug ---------------------
def eda_word_del(args,
        input_ids: torch.Tensor,
        input_emb: torch.Tensor,
        labels: torch.Tensor,
        eda_word_del_aug: torch.Tensor,
        eda_word_swap_aug: torch.Tensor,
        eda_word_del_swap_aug: torch.Tensor,
        bts: torch.Tensor,
        cbt: torch.Tensor,
        mag: torch.Tensor,
        tokenizer,
        model: nn.Module) -> torch.Tensor:
    if torch.cuda.device_count() > 1:
        Embedding = model.module.roberta.embeddings.word_embeddings
    else:
        Embedding = model.roberta.embeddings.word_embeddings

    aug_batch = eda_word_del_aug.cuda()

    return aug_batch, ste(Embedding(aug_batch), input_emb)


def eda_word_swap(args,
        input_ids: torch.Tensor,
        input_emb: torch.Tensor,
        labels: torch.Tensor,
        eda_word_del_aug: torch.Tensor,
        eda_word_swap_aug: torch.Tensor,
        eda_word_del_swap_aug: torch.Tensor,
        bts: torch.Tensor,
        cbt: torch.Tensor,
        mag: torch.Tensor,
        tokenizer,
        model: nn.Module) -> torch.Tensor:
    if torch.cuda.device_count() > 1:
        Embedding = model.module.roberta.embeddings.word_embeddings
    else:
        Embedding = model.roberta.embeddings.word_embeddings

    aug_batch = eda_word_swap_aug.cuda()

    return aug_batch, ste(Embedding(aug_batch), input_emb)


def eda_word_del_swap(args,
        input_ids: torch.Tensor,
        input_emb: torch.Tensor,
        labels: torch.Tensor,
        eda_word_del_aug: torch.Tensor,
        eda_word_swap_aug: torch.Tensor,
        eda_word_del_swap_aug: torch.Tensor,
        bts: torch.Tensor,
        cbt: torch.Tensor,
        mag: torch.Tensor,
        tokenizer,
        model: nn.Module) -> torch.Tensor:
    if torch.cuda.device_count() > 1:
        Embedding = model.module.roberta.embeddings.word_embeddings
    else:
        Embedding = model.roberta.embeddings.word_embeddings

    aug_batch = eda_word_del_swap_aug.cuda()

    return aug_batch, ste(Embedding(aug_batch), input_emb)


def backtrans(args,
        input_ids: torch.Tensor,
        input_emb: torch.Tensor,
        labels: torch.Tensor,
        eda_word_del_aug: torch.Tensor,
        eda_word_swap_aug: torch.Tensor,
        eda_word_del_swap_aug: torch.Tensor,
        bts: torch.Tensor,
        cbt: torch.Tensor,
        mag: torch.Tensor,
        tokenizer,
        model: nn.Module) -> torch.Tensor:
    if torch.cuda.device_count() > 1:
        Embedding = model.module.roberta.embeddings.word_embeddings
    else:
        Embedding = model.roberta.embeddings.word_embeddings

    aug_batch = bts.cuda()

    return aug_batch, ste(Embedding(aug_batch), input_emb)


def cbert(args,
        input_ids: torch.Tensor,
        input_emb: torch.Tensor,
        labels: torch.Tensor,
        eda_word_del_aug: torch.Tensor,
        eda_word_swap_aug: torch.Tensor,
        eda_word_del_swap_aug: torch.Tensor,
        bts: torch.Tensor,
        cbt: torch.Tensor,
        mag: torch.Tensor,
        tokenizer,
        model: nn.Module) -> torch.Tensor:
    if torch.cuda.device_count() > 1:
        Embedding = model.module.roberta.embeddings.word_embeddings
    else:
        Embedding = model.roberta.embeddings.word_embeddings

    aug_batch = cbt.cuda()

    return aug_batch, ste(Embedding(aug_batch), input_emb)


def random_mask(args,
        input_ids: torch.Tensor,
        input_emb: torch.Tensor,
        labels: torch.Tensor,
        eda_word_del_aug: torch.Tensor,
        eda_word_swap_aug: torch.Tensor,
        eda_word_del_swap_aug: torch.Tensor,
        bts: torch.Tensor,
        cbt: torch.Tensor,
        mag: torch.Tensor,
        tokenizer,
        model: nn.Module) -> torch.Tensor:
    if torch.cuda.device_count() > 1:
        Embedding = model.module.roberta.embeddings.word_embeddings
    else:
        Embedding = model.roberta.embeddings.word_embeddings
    
    aug_batch, _ = mask_tokens(
        inputs=input_ids,
        tokenizer=tokenizer,
        mask_probability = args.aug_mask_ratio
    )

    return aug_batch, ste(Embedding(aug_batch), input_emb)


# --------------------- Continuous Aug ---------------------
def r3f(args,
        input_ids: torch.Tensor,
        input_emb: torch.Tensor,
        labels: torch.Tensor,
        eda_word_del_aug: torch.Tensor,
        eda_word_swap_aug: torch.Tensor,
        eda_word_del_swap_aug: torch.Tensor,
        bts: torch.Tensor,
        cbt: torch.Tensor,
        mag: torch.Tensor,
        tokenizer,
        model: nn.Module) -> torch.Tensor:
    eps = mag.view(-1, 1, 1).cuda()
    attention_mask = (input_ids != 1).float().unsqueeze(2).cuda() #roberta [PAD] idx: 1

    noise_sampler = torch.distributions.uniform.Uniform(low=-1, high=1)
    noise = noise_sampler.sample(sample_shape=input_emb.shape).to(input_emb)
    noise = eps * noise * attention_mask  # to remove the noise on [PAD] mask

    return input_ids, input_emb + noise


def cutoff(args,
        input_ids: torch.Tensor,
        input_emb: torch.Tensor,
        labels: torch.Tensor,
        eda_word_del_aug: torch.Tensor,
        eda_word_swap_aug: torch.Tensor,
        eda_word_del_swap_aug: torch.Tensor,
        bts: torch.Tensor,
        cbt: torch.Tensor,
        mag: torch.Tensor,
        tokenizer,
        model: nn.Module) -> torch.Tensor:
    batch_size = input_ids.size(0)
    mag = mag.view(-1, 1, 1)
    attention_mask = (input_ids != 1).float()
    num_tokens = attention_mask.sum(dim=1, keepdim=True).cpu()

    # change of tensor is verified
    embed_cutoff = []
    for i in range(batch_size):
        cutoff_size = mag[0].data
        cutoff_length = int(num_tokens[i] * float(cutoff_size))
        start_idx = int(torch.rand(1) * (int(num_tokens[i]) - cutoff_length))
        cutoff_embed = torch.cat(
            (input_emb[i][:start_idx],
            torch.zeros([cutoff_length, input_emb.shape[-1]], dtype=torch.float).to(input_emb),
            input_emb[i][start_idx + cutoff_length:]), dim=0)
        embed_cutoff.append(cutoff_embed)
    embed_cutoff = torch.stack(embed_cutoff, dim=0)

    return input_ids, ste(embed_cutoff, mag.cuda())