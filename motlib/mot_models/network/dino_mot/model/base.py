# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from .my_dino import MyDINO
import torch
import copy
from torch import nn


class MOTDINOBASE(nn.Module):
    def init_dino(self, backbone, transformer):
        args = self.args
        num_classes = args.num_classes
        try:
            match_unstable_error = args.match_unstable_error
            dn_labelbook_size = args.dn_labelbook_size
        except:
            match_unstable_error = True
            dn_labelbook_size = num_classes

        try:
            dec_pred_class_embed_share = args.dec_pred_class_embed_share
        except:
            dec_pred_class_embed_share = True
        try:
            dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
        except:
            dec_pred_bbox_embed_share = True


        model = MyDINO(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=True,
            iter_update=True,
            query_dim=4,
            random_refpoints_xy=args.random_refpoints_xy,
            fix_refpoints_hw=args.fix_refpoints_hw,
            num_feature_levels=args.num_feature_levels,
            nheads=args.nheads,
            dec_pred_class_embed_share=dec_pred_class_embed_share,
            dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
            # two stage
            two_stage_type=args.two_stage_type,
            # box_share
            two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
            two_stage_class_embed_share=args.two_stage_class_embed_share,
            decoder_sa_type=args.decoder_sa_type,
            num_patterns=args.num_patterns,
            dn_number = args.dn_number if args.use_dn else 0,
            dn_box_noise_scale = args.dn_box_noise_scale,
            dn_label_noise_ratio = args.dn_label_noise_ratio,
            dn_labelbook_size = dn_labelbook_size,
        )

        
        return model
