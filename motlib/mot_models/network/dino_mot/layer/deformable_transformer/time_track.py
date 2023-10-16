# Copyright (2023) Bytedance Ltd. and/or its affiliates 


import torch
from torch import nn, Tensor
from typing import Optional
import torch.nn.functional as F
import copy
import numpy as np

from util.misc import inverse_sigmoid, scale_sigmoid
from .mot_deformable_transformer import MotDeformableTransformer
from .default_decoder import MotDeformableTransformerDecoderLayer, MotTransformerDecoder
from models.dino.ops.modules import MSDeformAttn
from models.dino.ops.functions import MSDeformAttnFunction
from models.dino.utils import MLP, gen_sineembed_for_position, _get_activation_fn
from motlib.mot_models.network.dino_mot.layer.self_attention.multihead_self_attention import MOTSelfAttention
from .default_decoder import MotDeformableTransformerDecoderLayer, MotTransformerDecoder

from . import DEFORMABLE_REGISTRY

__all__ = ['TimeTrackDeformableTransformer']


@DEFORMABLE_REGISTRY.register()
class TimeTrackDeformableTransformer(MotDeformableTransformer):
    def init_decoder(self, *args, **kwds):
        return TimeTrackV10TransformerDecoder(*args, **kwds)
    
    def init_decoder_layer(self, args, num_decoder_layers, d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points, use_deformable_box_attn, box_attn_type, key_aware_type, decoder_sa_type, module_seq):
        decoder_layer = nn.ModuleList([
        TimeTrackDeformableTransformerDecoderLayer(
            args, layer_id, d_model, dim_feedforward, dropout, activation,
            num_feature_levels, nhead, dec_n_points, use_deformable_box_attn=use_deformable_box_attn, box_attn_type=box_attn_type,
            key_aware_type=key_aware_type, decoder_sa_type=decoder_sa_type, module_seq=module_seq) for layer_id in range(num_decoder_layers)])

        return decoder_layer


class TimeTrackV10TransformerDecoder(MotTransformerDecoder):

    def get_track_instances_fea(self, track_instances):
        fea_list = []
        label_list = []
        ref_point_list = []
        for i in range(len(track_instances)):
            fea_num = len(track_instances.mem_bank[i])
            for i_fea, fea in enumerate(track_instances.mem_bank[i]):
                if i_fea < fea_num -1:
                    fea_list.append(fea)
                    label_list.append(i)
                    ref_point_list.append(track_instances.ref_bank[i][-1])
        
        if len(fea_list) == 0:
            return None, None, None, None
        
        fea_list = torch.stack(fea_list, dim=0).unsqueeze(1).detach()
        time_ref_point = torch.stack(ref_point_list, dim=0).unsqueeze(1)
        track_label = torch.arange(len(track_instances)).to(fea_list.device)

        pad_label = torch.asarray(label_list, dtype=track_label.dtype, device=track_label.device)

        tgt_mask = torch.cat((track_label, pad_label), dim=0)
        N = tgt_mask.shape[0]
        tgt_mask = tgt_mask.view(N, 1).expand(N, N).ne(tgt_mask.view(N, 1).expand(N, N).t())
        return tgt_mask, fea_list, time_ref_point, pad_label

    def prepare_for_time_track(self, track_instances, track_info, tgt, reference_points, tgt_mask):
        pad_size= track_info.get('pad_size', 0)
        det_query_num = track_info.get('det_query_num', 0)
        track_query_num = track_info.get('track_query_num', 0)
        assert len(track_instances) == track_query_num
        assert pad_size + det_query_num + track_query_num == tgt.shape[0]

        if track_query_num == 0:
            return tgt, reference_points, None
        
        time_mask, time_fea, time_refpoint, time_label = self.get_track_instances_fea(track_instances)
        if time_mask is None:
            return tgt, reference_points, None

        tgt = torch.cat((tgt,  time_fea), dim=0)
        reference_points = torch.cat((reference_points, time_refpoint), dim=0)

        time_num = time_label.shape[0]
        attn_mask = torch.ones(pad_size+det_query_num+track_query_num+time_num, pad_size+det_query_num+track_query_num+time_num).to('cuda') < 0
        
        # match query cannot see the reconstruct
        if tgt_mask is not None:
            attn_mask[:pad_size+det_query_num+track_query_num, :pad_size+det_query_num+track_query_num] = tgt_mask
        attn_mask[pad_size+det_query_num:, pad_size+det_query_num:] = ~time_mask
        attn_mask[pad_size:, :pad_size] = True
        attn_mask = attn_mask.to(torch.int)
        attn_mask = attn_mask - torch.diag_embed(torch.diag(attn_mask))
        attn_mask = attn_mask.to(torch.bool)

        return tgt, reference_points, {'time_mask': time_mask, 'time_label': time_label, 'pad_size': pad_size, 'det_query_num': det_query_num, 'track_query_num': track_query_num, 'attn_mask': attn_mask}

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None, # num_queries, bs, 2
                # for memory
                level_start_index: Optional[Tensor] = None, # num_levels
                spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None,
                track_instances = None,
                track_info = None
                ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt

        intermediate = []
        reference_points = scale_sigmoid(refpoints_unsigmoid.sigmoid())
        
        output, reference_points, other_input = self.prepare_for_time_track(track_instances, track_info, tgt, reference_points, tgt_mask)

        ref_points = [reference_points] 

        for layer_id, layer in enumerate(self.layers):

            # preprocess ref points
            if other_input is not None:
                if layer_id > self.args.inter_action_layer:
                    other_input['inter_act'] = True
                else:
                    other_input['inter_act'] = False
            output, reference_points, track_res_layer = self.forward_one_layer(layer_id=layer_id, layer=layer, output=output, memory=memory,reference_points=reference_points, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, level_start_index=level_start_index,spatial_shapes=spatial_shapes,valid_ratios=valid_ratios,pos=pos, tgt_mask=tgt_mask, memory_mask=memory_mask, ref_points=ref_points, intermediate=intermediate, other_input=other_input)
        
        if other_input is None:
            other_input = {}
        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
            other_input
        ]


class InfoCollectLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.self_attn_cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, tgt, tgt_query_pos, self_attn_mask):

        q = k = self.with_pos_embed(tgt, tgt_query_pos)
        tgt_cross = self.self_attn_cross(q, k, tgt, attn_mask=self_attn_mask)[0]

        q = tgt_cross
        k = v = tgt
        tgt_add = self.self_attn(q, k, v, attn_mask=self_attn_mask)[0]
        return tgt_add


class HistoryQIM(nn.Module):
    def __init__(self, d_model, n_heads, dropout, activation) -> None:
        super().__init__()

        self.head_dim = d_model // n_heads

        self.self_attn_add = InfoCollectLayer(d_model, n_heads, dropout=dropout)
        self.self_attn_drop = InfoCollectLayer(d_model, n_heads, dropout=dropout)

        self.activation_drop = _get_activation_fn(activation, d_model=d_model, batch_dim=1)
        self.dropout_drop = nn.Dropout(dropout)
        self.linear_drop = nn.Linear(d_model, n_heads)
    
        self.dropout_add = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, tgt, tgt_query_pos, self_attn_mask):

        tgt_add = self.self_attn_add(tgt, tgt_query_pos, self_attn_mask)

        r_drop = self.self_attn_drop(tgt, tgt_query_pos, self_attn_mask)
        r_drop = self.linear_drop(self.dropout_drop(self.activation_drop(r_drop))).sigmoid()
        seq_len, bs, n_heads = r_drop.shape
        r_drop = r_drop.unsqueeze(dim=-1).expand((seq_len, bs, n_heads, self.head_dim)).reshape((seq_len, bs, -1))

        tgt = tgt * (1 - r_drop) * 2 + self.dropout_add(tgt_add)
        tgt = self.norm(tgt)
        return tgt


class TimeTrackDeformableTransformerDecoderLayer(MotDeformableTransformerDecoderLayer):
    def __init__(self, args, layer_id, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_levels=4, n_heads=8, n_points=4, use_deformable_box_attn=False, box_attn_type='roi_align', key_aware_type=None, decoder_sa_type='ca', module_seq=...):
        super().__init__(args, layer_id, d_model, d_ffn, dropout, activation, n_levels, n_heads, n_points, use_deformable_box_attn, box_attn_type, key_aware_type, decoder_sa_type, module_seq)

        self.history_query_ia_flag = self.layer_id > self.args.inter_action_layer

        if self.history_query_ia_flag:
            self.history_qim = HistoryQIM(d_model=d_model, n_heads=n_heads, dropout=dropout, activation=activation)
    
    def forward_sa_time(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                other_input = None
            ):
     
        if not self.history_query_ia_flag:
            return tgt
        
        if other_input is None or not other_input['inter_act']:
            tgt = tgt + self.history_qim(tgt[:3], tgt_query_pos[:3], None).mean() * 0
            return tgt
        pad_size = other_input['pad_size']
        det_query_num = other_input['det_query_num']

        tgt_track = tgt[pad_size+det_query_num:]
        tgt_query_pos_track = tgt_query_pos[pad_size+det_query_num:]

        self_attn_mask = other_input['time_mask']

        tgt_track = self.history_qim(tgt_track, tgt_query_pos_track, self_attn_mask)

        tgt = torch.cat((tgt[:pad_size+det_query_num], tgt_track), dim=0)
        
        return tgt
    
    def forward_sa_cross(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                other_input: Optional[Tensor] = None, # mask used for cross-attention
            ):

        if other_input is not None:
            attn_mask = other_input['attn_mask']

        else:
            attn_mask = self_attn_mask

        q = k = self.with_pos_embed(tgt, tgt_query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=attn_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt
    
    def forward(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
                other_input = None
            ):
        
        tgt = self.forward_sa_time(tgt=tgt, tgt_query_pos=tgt_query_pos, other_input=other_input)

        for funcname in self.module_seq:
            if funcname == 'ffn':
                tgt = self.forward_ffn(tgt)
            elif funcname == 'ca':
                tgt = self.forward_ca(tgt, tgt_query_pos, tgt_query_sine_embed, \
                    tgt_key_padding_mask, tgt_reference_points, \
                        memory, memory_key_padding_mask, memory_level_start_index, \
                            memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            elif funcname == 'sa':
                tgt = self.forward_sa_cross(tgt, tgt_query_pos, self_attn_mask, other_input)
                
            else:
                raise ValueError('unknown funcname {}'.format(funcname))

        return tgt