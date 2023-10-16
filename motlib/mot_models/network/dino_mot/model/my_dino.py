# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from models.dino.dino import DINO

import copy
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid, scale_sigmoid)

from models.dino.utils import sigmoid_focal_loss, MLP
from models.dino.dn_components import prepare_for_cdn, dn_post_process
from motlib.mot_models.structures import Instances
from motlib.utils.timer import Timer


class MyDINO(DINO):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, iter_update=False, query_dim=2, random_refpoints_xy=False, fix_refpoints_hw=-1, num_feature_levels=1, nheads=8, two_stage_type='no', two_stage_add_query_num=0, dec_pred_class_embed_share=True, dec_pred_bbox_embed_share=True, two_stage_class_embed_share=True, two_stage_bbox_embed_share=True, decoder_sa_type='sa', num_patterns=0, dn_number=100, dn_box_noise_scale=0.4, dn_label_noise_ratio=0.5, dn_labelbook_size=100):
        super().__init__(backbone, transformer, num_classes, num_queries, aux_loss, iter_update, query_dim, random_refpoints_xy, fix_refpoints_hw, num_feature_levels, nheads, two_stage_type, two_stage_add_query_num, dec_pred_class_embed_share, dec_pred_bbox_embed_share, two_stage_class_embed_share, two_stage_bbox_embed_share, decoder_sa_type, num_patterns, dn_number, dn_box_noise_scale, dn_label_noise_ratio, dn_labelbook_size)

        self.timer_resnet = Timer()
        self.timer_encoder = Timer()
        self.timer_decoder = Timer()
        self.timer_tracking = Timer()
        self.timer_resnet.pause()
        self.timer_encoder.pause()
        self.timer_decoder.pause()
        self.timer_tracking.pause()

    def forward_encoder(self, samples):

        if not self.training:
            self.timer_resnet.resume()

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)
        
        if not self.training:
            self.timer_resnet.pause()
            self.timer_encoder.resume()

        memory, mask_flatten, spatial_shapes, lvl_pos_embed_flatten, level_start_index, valid_ratios, bs = self.transformer.forward_encoder(srcs, masks, poss)

        if not self.training:
            self.timer_encoder.pause()
        
        return memory, mask_flatten, spatial_shapes, lvl_pos_embed_flatten, level_start_index, valid_ratios, bs

    def prepare_cdn(self, targets, track_instances_num=0):
        if self.dn_number > 0 or targets is not None:
            num_queries = self.num_queries + track_instances_num
            input_query_label, input_query_bbox, attn_mask, dn_meta =\
                prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                                training=self.training,num_queries=num_queries,num_classes=self.num_classes,
                                hidden_dim=self.hidden_dim,label_enc=self.label_enc)
        else:
            assert targets is None
            input_query_bbox = input_query_label = attn_mask = dn_meta = None
        
        return input_query_label, input_query_bbox, attn_mask, dn_meta
    
    def process_encoder_output(self, hs_enc, ref_enc, out, init_box_proposal):
        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

            # prepare enc outputs
            # import ipdb; ipdb.set_trace()
            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    layer_enc_outputs_coord = scale_sigmoid(layer_enc_outputs_coord_unsig.sigmoid())

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                # enc_delta_unsig = self.enc_bbox_embed(hs_enc[:-1])
                # enc_outputs_unsig = enc_delta_unsig + ref_enc[:-1]
                # enc_outputs_coord = enc_outputs_unsig.sigmoid()
                # enc_outputs_class = self.enc_class_embed(hs_enc[:-1])
                out['enc_outputs'] = [
                    {'pred_logits': a, 'pred_boxes': b} for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]
        return out
        

    def forward(self, memory, mask_flatten, spatial_shapes, lvl_pos_embed_flatten, level_start_index, valid_ratios, bs, targets:List=None, track_instances:Instances=None):

        track_instances_num = len(track_instances) 

        input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_cdn(targets, track_instances_num)

        hs, reference, hs_enc, ref_enc, init_box_proposal, track_dec_res = self.transformer(
            memory, mask_flatten, spatial_shapes, lvl_pos_embed_flatten, level_start_index, valid_ratios, bs, input_query_bbox, input_query_label, attn_mask, 
            track_instances
            )
        # In case num object=0
        hs[0]+=self.label_enc.weight[0,0]*0.0

        # deformable-detr-like anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig  + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = scale_sigmoid(layer_outputs_unsig.sigmoid())
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)        


        # outputs_class = self.class_embed(hs)
        outputs_class = torch.stack([layer_cls_embed(layer_hs) for
                                     layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list = \
                dn_post_process(outputs_class, outputs_coord_list,
                                dn_meta,self.aux_loss,self._set_aux_loss)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)
        
        out = self.process_encoder_output(hs_enc, ref_enc, out, init_box_proposal)

        out['dn_meta'] = dn_meta
        out['hs_last_layer'] = hs[-1]
        out['track_dec_out'] = track_dec_res

        return out
