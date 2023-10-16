import torch
import torch.nn as nn
from torch.nn import functional as F
from models.dino.utils import MLP
import math

class MOTSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()

        self.num_attention_heads = num_heads
        self.attention_head_size = embed_dim // num_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = MLP(embed_dim, embed_dim,  self.all_head_size, 1)
        self.key = MLP(embed_dim, embed_dim,  self.all_head_size, 1)
        self.value = MLP(embed_dim, embed_dim,  self.all_head_size, 2)

        # self.dense = nn.Linear(embed_dim, embed_dim)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape).permute(1, 2, 0, 3)   # [Batch_size, Num_of_heads, Seq_length, Head_size]

        return x.view((-1,)+x.shape[-2:])
    
    def transpose_for_value(self, x):
        new_x_shape = (1,) + x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 3, 1, 2, 4)

    def forward(self, q, k, v, attn_mask):
        bs = q.shape[1]
        mixed_query_layer = self.query(q)  
        mixed_key_layer = self.key(k) 
        mixed_value_layer = self.value(v) 

        query_layer = self.transpose_for_scores(
            mixed_query_layer)  # [Batch_size x Num_of_heads,  Seq_length, Head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [Batch_size x Num_of_heads, Seq_length, Head_size]
        value_layer = self.transpose_for_value(
            mixed_value_layer)  # [Batch_size x Num_of_heads x Seq_length x Head_size]

        # query_layer = query_layer / math.sqrt(
        #     self.attention_head_size)  

        query_layer = F.normalize(query_layer, p=2, dim=-1)
        key_layer = F.normalize(key_layer, p=2, dim=-1)

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
        
        if attn_mask is not None:
            attention_scores = torch.baddbmm(attn_mask, query_layer, key_layer.transpose(-2, -1))
        else:
            attention_scores = torch.bmm(query_layer, key_layer.transpose(-2, -1))

        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_probs = attention_probs.view((bs, self.num_attention_heads) + attention_probs.shape[1:])
        context_layer = attention_probs[..., None] * value_layer
        context_layer = context_layer.sum(dim=-2)

        context_layer = context_layer.permute(2, 0, 1,
                                              3).contiguous() 
        new_context_layer_shape = context_layer.size()[:-2] + (
        self.all_head_size,) 
        output = context_layer.view(*new_context_layer_shape) 

        # output = self.dense(output)

        return output, mixed_query_layer, mixed_key_layer