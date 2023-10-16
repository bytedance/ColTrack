# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from models.dino.deformable_transformer import DeformableTransformer, DeformableTransformerEncoderLayer, DeformableTransformerDecoderLayer, TransformerEncoder, TransformerDecoder
from models.dino.utils import gen_encoder_output_proposals
from .default_decoder import MotDeformableTransformerDecoderLayer, MotTransformerDecoder
import torch
import copy
from torch import nn, Tensor
from util.misc import inverse_sigmoid, scale_sigmoid

from . import DEFORMABLE_REGISTRY

__all__ = ['MotDeformableTransformer']


@DEFORMABLE_REGISTRY.register()
class MotDeformableTransformer(DeformableTransformer):
    def __init__(self, args, d_model=256, nhead=8, 
                 num_queries=300, 
                 num_encoder_layers=6,
                 num_unicoder_layers=0,
                 num_decoder_layers=6, 
                 dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=4,
                 num_patterns=0,
                 modulate_hw_attn=False,
                 # for deformable encoder
                 deformable_encoder=False,
                 deformable_decoder=False,
                 num_feature_levels=1,
                 enc_n_points=4,
                 dec_n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 # init query
                 learnable_tgt_init=False,
                 decoder_query_perturber=None,
                 add_channel_attention=False,
                 add_pos_value=False,
                 random_refpoints_xy=False,
                 # two stage
                 two_stage_type='no', # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
                 two_stage_pat_embed=0,
                 two_stage_add_query_num=0,
                 two_stage_learn_wh=False,
                 two_stage_keep_all_tokens=False,
                 # evo of #anchors
                 dec_layer_number=None,
                 rm_enc_query_scale=True,
                 rm_dec_query_scale=True,
                 rm_self_attn_layers=None,
                 key_aware_type=None,
                 # layer share
                 layer_share_type=None,
                 # for detach
                 rm_detach=None,
                 decoder_sa_type='ca', 
                 module_seq=['sa', 'ca', 'ffn'],
                 # for dn
                 embed_init_tgt=False,

                 use_detached_boxes_dec_out=False,
                 ):
        super(DeformableTransformer, self).__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.deformable_encoder = deformable_encoder
        self.deformable_decoder = deformable_decoder
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_queries = num_queries
        self.random_refpoints_xy = random_refpoints_xy
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        assert query_dim == 4

        if num_feature_levels > 1:
            assert deformable_encoder, "only support deformable_encoder for num_feature_levels > 1"
        if use_deformable_box_attn:
            assert deformable_encoder or deformable_encoder

        assert layer_share_type in [None, 'encoder', 'decoder', 'both']
        if layer_share_type in ['encoder', 'both']:
            enc_layer_share = True
        else:
            enc_layer_share = False
        if layer_share_type in ['decoder', 'both']:
            dec_layer_share = True
        else:
            dec_layer_share = False
        assert layer_share_type is None

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        # choose encoder layer type
        if deformable_encoder:
            encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points, add_channel_attention=add_channel_attention, use_deformable_box_attn=use_deformable_box_attn, box_attn_type=box_attn_type)
        else:
            raise NotImplementedError
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, 
            encoder_norm, d_model=d_model, 
            num_queries=num_queries,
            deformable_encoder=deformable_encoder, 
            enc_layer_share=enc_layer_share, 
            two_stage_type=two_stage_type
        )

        # choose decoder layer type
        if deformable_decoder:
            decoder_layer = self.init_decoder_layer(args, num_decoder_layers, d_model, dim_feedforward,
                                                    dropout, activation,
                                                    num_feature_levels, nhead, dec_n_points, use_deformable_box_attn, box_attn_type,
                                                    key_aware_type, decoder_sa_type, module_seq)

        else:
            raise NotImplementedError

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = self.init_decoder(args, decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec,
                                        d_model=d_model, query_dim=query_dim, 
                                        modulate_hw_attn=modulate_hw_attn,
                                        num_feature_levels=num_feature_levels,
                                        deformable_decoder=deformable_decoder,
                                        decoder_query_perturber=decoder_query_perturber, 
                                        dec_layer_number=dec_layer_number, rm_dec_query_scale=rm_dec_query_scale,
                                        dec_layer_share=dec_layer_share,
                                        use_detached_boxes_dec_out=use_detached_boxes_dec_out
                                        )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries # useful for single stage model only
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None
        
        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != 'no' and embed_init_tgt) or (two_stage_type == 'no'):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None
            
        # for two stage
        self.two_stage_type = two_stage_type
        self.two_stage_pat_embed = two_stage_pat_embed
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type =='standard':
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)      
            
            if two_stage_pat_embed > 0:
                self.pat_embed_for_2stage = nn.Parameter(torch.Tensor(two_stage_pat_embed, d_model))
                nn.init.normal_(self.pat_embed_for_2stage)

            if two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(self.two_stage_add_query_num, d_model)

            if two_stage_learn_wh:
                # import ipdb; ipdb.set_trace()
                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None

        if two_stage_type == 'no':
            self.init_ref_points(num_queries) # init self.refpoint_embed


        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        # evolution of anchors
        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            if self.two_stage_type != 'no' or num_patterns == 0:
                assert dec_layer_number[0] == num_queries, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries})"
            else:
                assert dec_layer_number[0] == num_queries * num_patterns, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries}) * num_patterns({num_patterns})"

        self._reset_parameters()

        self.rm_self_attn_layers = rm_self_attn_layers
        if rm_self_attn_layers is not None:
            # assert len(rm_self_attn_layers) == num_decoder_layers
            print("Removing the self-attn in {} decoder layers".format(rm_self_attn_layers))
            for lid, dec_layer in enumerate(self.decoder.layers):
                if lid in rm_self_attn_layers:
                    dec_layer.rm_self_attn_modules()

        self.rm_detach = rm_detach
        if self.rm_detach:
            assert isinstance(rm_detach, list)
            assert any([i in ['enc_ref', 'enc_tgt', 'dec'] for i in rm_detach])
        self.decoder.rm_detach = rm_detach
    
    def init_decoder(self, *args, **kwds):
        return MotTransformerDecoder(*args, **kwds)
    
    def init_decoder_layer(self, args, num_decoder_layers, d_model, dim_feedforward,
        dropout, activation,
        num_feature_levels, nhead, dec_n_points, use_deformable_box_attn, box_attn_type,
        key_aware_type,
        decoder_sa_type,
        module_seq):
        decoder_layer = nn.ModuleList([
        MotDeformableTransformerDecoderLayer(
            args, layer_id, d_model, dim_feedforward, dropout, activation,
            num_feature_levels, nhead, dec_n_points, use_deformable_box_attn=use_deformable_box_attn, box_attn_type=box_attn_type,
            key_aware_type=key_aware_type, decoder_sa_type=decoder_sa_type, module_seq=module_seq) for layer_id in range(num_decoder_layers)])
        return decoder_layer

    def forward_encoder(self, srcs, masks, pos_embeds):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            mask = mask.flatten(1)                              # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c 
        mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # bs, \sum{hxw}, c 
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None

        #########################################################
        # Begin Encoder
        #########################################################
        memory, enc_intermediate_output, enc_intermediate_refpoints = self.encoder(
                src_flatten, 
                pos=lvl_pos_embed_flatten, 
                level_start_index=level_start_index, 
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                key_padding_mask=mask_flatten,
                ref_token_index=enc_topk_proposals, # bs, nq 
                ref_token_coord=enc_refpoint_embed, # bs, nq, 4
                )
        return memory, mask_flatten, spatial_shapes, lvl_pos_embed_flatten, level_start_index, valid_ratios, bs
        
    def prepare_for_decoder(self, memory, mask_flatten, spatial_shapes, bs):
        if self.two_stage_learn_wh:
            input_hw = self.two_stage_wh_embedding.weight[0]
        else:
            input_hw = None
        output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes, input_hw)
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        
        if self.two_stage_pat_embed > 0:
            raise NotImplementedError

        if self.two_stage_add_query_num > 0:
            raise NotImplementedError

        enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
        enc_outputs_coord_unselected = self.enc_out_bbox_embed(output_memory) + output_proposals # (bs, \sum{hw}, 4) unsigmoid
        topk = self.num_queries
        topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1] # bs, nq
        

        # gather boxes
        refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid
        refpoint_embed_ = refpoint_embed_undetach.detach()
        init_box_proposal = scale_sigmoid(torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()) # sigmoid

        # gather tgt
        tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
        if self.embed_init_tgt:
            tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1) # nq, bs, d_model
        else:
            tgt_ = tgt_undetach.detach()
        
        return refpoint_embed_, tgt_, init_box_proposal, tgt_undetach, refpoint_embed_undetach
    
    def forward(self, memory, mask_flatten, spatial_shapes, lvl_pos_embed_flatten, level_start_index, valid_ratios, bs, refpoint_embed, tgt, attn_mask=None, track_instances=None):

        track_instances_num = len(track_instances)
        track_query = track_instances.query_pos if track_instances_num > 0 else None
        track_ref_pts = track_instances.ref_pts if track_instances_num > 0 else None

        track_res = {}

        if self.two_stage_type =='standard':
            refpoint_embed_, tgt_, init_box_proposal, tgt_undetach, refpoint_embed_undetach = self.prepare_for_decoder(memory, mask_flatten, spatial_shapes, bs)
            refpoint_embed_expand = []
            tgt_expand = []

            if refpoint_embed is not None:
                refpoint_embed_expand.append(refpoint_embed)
                tgt_expand.append(tgt)
                track_res['pad_size'] = tgt.shape[1]
            refpoint_embed_expand.append(refpoint_embed_)
            tgt_expand.append(tgt_)
            track_res['det_query_num'] = tgt_.shape[1]
            if track_ref_pts is not None:
                track_res['track_query_num'] = track_ref_pts.shape[0]
                refpoint_embed_expand.append(track_ref_pts[None, :, :].repeat(bs, 1, 1))
                tgt_expand.append(track_query[None, :, :].repeat(bs, 1, 1))
            
            if len(refpoint_embed_expand) == 1:
                refpoint_embed,tgt=refpoint_embed_,tgt_
            else:
                refpoint_embed=torch.cat(refpoint_embed_expand,dim=1)
                tgt=torch.cat(tgt_expand,dim=1)
        else:
            raise NotImplementedError("unknown two_stage_type {}".format(self.two_stage_type))
        #########################################################
        # End preparing tgt
        # - tgt: bs, NQ, d_model
        # - refpoint_embed(unsigmoid): bs, NQ, d_model 
        ######################################################### 


        #########################################################
        # Begin Decoder
        #########################################################
        hs, references, track_dec_out = self.decoder(
                tgt=tgt.transpose(0, 1), 
                memory=memory.transpose(0, 1), 
                memory_key_padding_mask=mask_flatten, 
                pos=lvl_pos_embed_flatten.transpose(0, 1),
                refpoints_unsigmoid=refpoint_embed.transpose(0, 1), 
                level_start_index=level_start_index, 
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,tgt_mask=attn_mask, track_instances=track_instances, track_info=track_res)
        
        assert isinstance(track_dec_out, dict)
        track_res['decoder_out'] = track_dec_out
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        #########################################################


        #########################################################
        # Begin postprocess
        #########################################################     
        if self.two_stage_type == 'standard':
            if self.two_stage_keep_all_tokens:
                raise NotImplementedError
                hs_enc = output_memory.unsqueeze(0)
                ref_enc = enc_outputs_coord_unselected.unsqueeze(0)
                init_box_proposal = output_proposals
                # import ipdb; ipdb.set_trace()
            else:
                hs_enc = tgt_undetach.unsqueeze(0)
                ref_enc = scale_sigmoid(refpoint_embed_undetach.sigmoid()).unsqueeze(0)
        else:
            hs_enc = ref_enc = None
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################        

        return hs, references, hs_enc, ref_enc, init_box_proposal, track_res