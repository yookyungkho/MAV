# Code is mainly adopted from https://github.com/moskomule/dda/tree/fasteraa/faster_autoaugment
from __future__ import annotations

import json
import random
from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.distributions import Categorical

from .operations import *

def load_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data

def save_json(file_path, save_file):
    json.dump(save_file, open(file_path,'w'), indent=4)


# ---------------------- Auto Augmentation (DND) ----------------------
class AutoAugPolicy(nn.Module):
    def __init__(self,
                temperature: float,
                record_path: str,
                ):
        super(AutoAugPolicy, self).__init__()
        self.operations = nn.ModuleList(self.eda_operations())

        self.op_names = [
            "EDA_WordDelete", "EDA_WordSwap", "EDA_WordDelete_Swap",
            "Cbert", "BackTrans", "RandomMask", "R3F", "Cutoff"
        ]
        self.record_path = record_path
        self.create_op_records(self.record_path)

        self._weights = nn.Parameter(torch.ones(len(self.operations)))
        self.temperature = temperature


    @staticmethod
    def eda_operations():
        return [
            EDA_WordDelete(),
            EDA_WordSwap(),
            EDA_WordDelete_Swap(),
            Cbert(),
            BackTrans(),
            RandomMask(),
            R3F(),
            Cutoff()
        ]

    @property
    def weights(self
                ):
        return self._weights.div(self.temperature).softmax(0).cuda()

    def forward(self,
                args,
                input_ids: Tensor,
                input_emb: Tensor,
                labels: Tensor,
                eda_word_del_aug: Tensor,
                eda_word_swap_aug: Tensor,
                eda_word_del_swap_aug: Tensor,
                bts: torch.Tensor,
                cbt: torch.Tensor,
                tokenizer,
                model: nn.Module
                ) -> Tensor:
        if self.training:
            sampled_op = F.gumbel_softmax(self._weights, tau=self.temperature, hard=True).cuda()
            if self._weights.data.mean() == 1:
                sampled_idx = int(torch.randint(0, len(self.operations), (1,)))
            else:
                sampled_idx = torch.max(sampled_op, dim=0)[1]

            aug_name = self.op_names[sampled_idx]
            self.update_op_records(self.record_path, aug_name)
            print(f"\n ***** self.training - selected aug: {aug_name}")

            input_ids, input_emb = self.operations[sampled_idx](
                args, input_ids, input_emb, labels, eda_word_del_aug, eda_word_swap_aug, eda_word_del_swap_aug, bts, cbt, tokenizer, model
            )

            return input_ids, (input_emb.unsqueeze(0) * sampled_op.view(-1, 1, 1, 1)).sum(0)
        else:
            sampled_idx = Categorical(self.weights).sample()
            print(f"\n ***** not self.training: {sampled_idx}")
            return self.operations[sampled_idx](
                args, input_ids, input_emb, labels, eda_word_del_aug, eda_word_swap_aug, eda_word_del_swap_aug, bts, cbt, tokenizer, model
            )

    def create_op_records(self, record_path):
        op_dict = {'selected_augs': []}
        save_json(record_path, op_dict)
    
    def update_op_records(self, record_path, aug_name):
        op_dict = load_json(record_path)
        op_dict['selected_augs'].append(aug_name)
        save_json(record_path, op_dict)


# ---------------------- Random Augmentation ----------------------
class RandomAugment(nn.Module):
    def __init__(self, record_path):
        super(RandomAugment, self).__init__()
        self.operations = nn.ModuleList(self.eda_operations())
        self.op_names = [
            "EDA_WordDelete", "EDA_WordSwap", "EDA_WordDelete_Swap",
            "Cbert", "BackTrans", "RandomMask", "R3F", "Cutoff"
        ]
        self.record_path = record_path
        self.op_records = self.create_op_records(self.record_path)
        self._weights = nn.Parameter(torch.ones(len(self.operations)))

    @staticmethod
    def eda_operations():
        return [
            EDA_WordDelete(),
            EDA_WordSwap(),
            EDA_WordDelete_Swap(),
            Cbert(),
            BackTrans(),
            RandomMask(),
            R3F(),
            Cutoff()
        ]

    def forward(self,
                args,
                input_ids: Tensor,
                input_emb: Tensor,
                labels: Tensor,
                eda_word_del_aug: Tensor,
                eda_word_swap_aug: Tensor,
                eda_word_del_swap_aug: Tensor,
                bts: torch.Tensor,
                cbt: torch.Tensor,
                tokenizer,
                model: nn.Module
                ) -> Tensor:
        if self.training:
            sampled_idx = int(torch.randint(0, len(self.operations), (1,)))
            aug_name = self.op_names[sampled_idx]
            self.update_op_records(self.record_path, aug_name)
            print(f"\n ***** random aug: {aug_name}")

            input_ids, input_emb = self.operations[sampled_idx](
                args, input_ids, input_emb, labels, eda_word_del_aug, eda_word_swap_aug, eda_word_del_swap_aug, bts, cbt, tokenizer, model
            )
            
            return input_ids, input_emb
        else:
            sampled_idx = Categorical(self._weights).sample()
            print(f"\n ***** not self.training: {sampled_idx}")
            return self.operations[sampled_idx](
                args, input_ids, input_emb, labels, eda_word_del_aug, eda_word_swap_aug, eda_word_del_swap_aug, bts, cbt, tokenizer, model
            )
    
    def create_op_records(self, record_path):
        op_dict = {'selected_augs': []}
        save_json(record_path, op_dict)
    
    def update_op_records(self, record_path, aug_name):
        op_dict = load_json(record_path)
        op_dict['selected_augs'].append(aug_name)
        save_json(record_path, op_dict)


# ---------------------- Single Augmentation ----------------------
class SingleAugment(nn.Module):
    def __init__(self, aug_idx):
        super(SingleAugment, self).__init__()
        self.operations = nn.ModuleList(self.eda_operations())
        self.op_names = [
            "EDA_WordDelete", "EDA_WordSwap", "EDA_WordDelete_Swap",
            "Cbert", "BackTrans", "RandomMask", "R3F", "Cutoff"
        ]
        self.aug_idx = aug_idx

    @staticmethod
    def eda_operations():
        return [
            EDA_WordDelete(),
            EDA_WordSwap(),
            EDA_WordDelete_Swap(),
            Cbert(),
            BackTrans(),
            RandomMask(),
            R3F(),
            Cutoff()
        ]

    def forward(self,
                args,
                input_ids: Tensor,
                input_emb: Tensor,
                labels: Tensor,
                eda_word_del_aug: Tensor,
                eda_word_swap_aug: Tensor,
                eda_word_del_swap_aug: Tensor,
                bts: torch.Tensor,
                cbt: torch.Tensor,
                tokenizer,
                model: nn.Module
                ) -> Tensor:
        if self.training:
            input_ids, input_emb = self.operations[self.aug_idx](
                args, input_ids, input_emb, labels, eda_word_del_aug, eda_word_swap_aug, eda_word_del_swap_aug, bts, cbt, tokenizer, model
            )
            aug_name = self.op_names[self.aug_idx]
            return input_ids, input_emb
        else:
            sampled_idx = Categorical(self._weights).sample()
            print(f"\n ***** not self.training: {sampled_idx}")
            return self.operations[sampled_idx](
                args, input_ids, input_emb, labels, eda_word_del_aug, eda_word_swap_aug, eda_word_del_swap_aug, bts, cbt, tokenizer, model
            )
