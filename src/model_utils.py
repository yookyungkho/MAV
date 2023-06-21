import torch
from torch import nn

class MAV(nn.Module):
    """Mapping-free Automatic Verbalizer"""
    def __init__(self, config):
        super().__init__()
        self.vocab_extractor = nn.Linear(config.vocab_size, config.mav_hidden_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.mav_hidden_dim, config.num_labels)

    def forward(self, features, **kwargs):
        # FCL 1 (Vocab Extractor)
        x = torch.tanh(features)
        x = self.dropout(x)
        x = self.vocab_extractor(x)
        # FCL 2 (Output Layer)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class MaskClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # features: [MASK] representation
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
