import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

class TrackAtten(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.update_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.norm4 = nn.LayerNorm(d_model)

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        raise NotImplementedError