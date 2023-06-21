import torch
import json
import numpy as np

import torch.optim as optim
from transformers import PreTrainedTokenizer
from typing import Tuple

from .policy import AutoAugPolicy


def get_embed(model, input_ids):
    if torch.cuda.device_count() > 1:
        embed = model.module.roberta.embeddings.word_embeddings(input_ids)
    else:
        embed = model.roberta.embeddings.word_embeddings(input_ids)
    return embed


def load_augment(aug_data_dir):
    eda_word_del_src = np.load(f"{aug_data_dir}/unlabeled_worddelete.npy")
    eda_word_swap_src = np.load(f"{aug_data_dir}/unlabeled_wordswap.npy")
    eda_word_del_swap_src = np.load(f"{aug_data_dir}/unlabeled_worddelete*wordswap.npy")
    bts_src = np.load(f"{aug_data_dir}/unlabeled_backtranslation.npy")
    cbt_src = np.load(f"{aug_data_dir}/unlabeled_bertaug.npy")

    return (
        torch.LongTensor(eda_word_del_src),
        torch.LongTensor(eda_word_swap_src),
        torch.LongTensor(eda_word_del_swap_src),
        torch.LongTensor(bts_src),
        torch.LongTensor(cbt_src)
        )

def set_policy(temperature, policy_lr, record_path):
    policy = AutoAugPolicy(temperature, record_path)
    policy_optimizer = optim.Adam(policy.parameters(), lr=policy_lr)

    return policy, policy_optimizer


class PolicyParamSaver :
    def __init__(self, policy) :
        self.policy_grad_save = {
            "weights" : torch.zeros_like(policy._weights),
            "operation_mags" : torch.zeros((len(policy.operations), 1)),
            "operation_prob" : torch.zeros((len(policy.operations), 1))
        }
    def is_not_none(self, x) :
        return x is not None

    def save_policy_grad(self, loss_policy, policy) :
        self.policy_grad_save["weights"] += torch.autograd.grad(loss_policy, policy._weights, retain_graph=True)[0]
        mag_grads = [torch.autograd.grad(loss_policy, op._magnitude, retain_graph=True, allow_unused=True)[0] for op in policy.operations]
        prob_grads = [torch.autograd.grad(loss_policy, op._probability, retain_graph=True, allow_unused=True)[0] for op in policy.operations]
        self.policy_grad_save["operation_mags"] += torch.tensor([grad if grad is not None else 0 for grad in mag_grads]).unsqueeze(1)
        self.policy_grad_save["operation_prob"] +=  torch.tensor([grad if grad is not None else 0 for grad in prob_grads]).unsqueeze(1)

    def zero_grad_saved(self) :
        self.policy_grad_save["weights"].zero_()
        for op_mag in self.policy_grad_save["operation_mags"] :
                op_mag.zero_()
        for op_prob in self.policy_grad_save["operation_prob"] :
                op_prob.zero_()
    
    def zero_grad_policy(self, policy) :
        policy._weights.grad = torch.zeros_like(policy._weights)
        for op in policy.operations :
            if self.is_not_none(op._magnitude.grad) :
                op._magnitude.grad = torch.zeros_like(op._magnitude)
            if self.is_not_none(op._probability.grad) :
                op._probability.grad = torch.zeros_like(op._probability)

    def update_policy_grad(self, policy) :
        self.zero_grad_policy(policy)
        policy._weights.grad = self.policy_grad_save["weights"]
        for op, op_mag_grad, op_prob_grad in zip(policy.operations, self.policy_grad_save["operation_mags"], self.policy_grad_save["operation_prob"]) :
            if op_mag_grad != 0 :
                op._magnitude.grad = op_mag_grad
            else : 
                op._magnitude.grad = None # 0 means no gradient

            if op_prob_grad != 0 :
                    op._probability.grad = torch.tensor([op_prob_grad])
            else :
                op._probability.grad = None # 0 means no gradient
