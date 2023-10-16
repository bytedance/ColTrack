# Copyright (2023) Bytedance Ltd. and/or its affiliates 


from turtle import forward
from models.dino.deformable_transformer import DeformableTransformerDecoderLayer, TransformerDecoder
from models.dino.utils import gen_encoder_output_proposals, MLP,_get_activation_fn, gen_sineembed_for_position
from models.dino.ops.modules import MSDeformAttn
import torch
from torch import nn, Tensor
import math, random
from typing import Optional
from util.misc import inverse_sigmoid, scale_sigmoid


class MotTransformerDecoder(TransformerDecoder):
    def __init__(self, args, decoder_layer, num_layers, norm=None, return_intermediate=False, d_model=256, query_dim=4, modulate_hw_attn=False, num_feature_levels=1, deformable_decoder=False, decoder_query_perturber=None, dec_layer_number=None, rm_dec_query_scale=False, dec_layer_share=False, dec_layer_dropout_prob=None, use_detached_boxes_dec_out=False):
        super(TransformerDecoder, self).__init__()
        self.args = args
        self.layers = decoder_layer
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out

        
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers
            # assert dec_layer_number[0] == 
            
        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.rm_detach = None
        self.init()
    
    def init(self):
        pass
    
    def forward_one_layer(self, layer_id, layer, output, memory, reference_points, tgt_key_padding_mask, memory_key_padding_mask, level_start_index, spatial_shapes, valid_ratios, pos, tgt_mask, memory_mask, ref_points, intermediate, other_input=None):
        if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(reference_points)

        if self.deformable_decoder:
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                        * torch.cat([valid_ratios, valid_ratios], -1)[None, :] # nq, bs, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # nq, bs, 256*2 
        else:
            query_sine_embed = gen_sineembed_for_position(reference_points) # nq, bs, 256*2
            reference_points_input = None

        # conditional query
        # import ipdb; ipdb.set_trace()
        raw_query_pos = self.ref_point_head(query_sine_embed) # nq, bs, 256
        pos_scale = self.query_scale(output) if self.query_scale is not None else 1
        query_pos = pos_scale * raw_query_pos
        if not self.deformable_decoder:
            query_sine_embed = query_sine_embed[..., :self.d_model] * self.query_pos_sine_scale(output)

        # modulated HW attentions
        if not self.deformable_decoder and self.modulate_hw_attn:
            refHW_cond = scale_sigmoid(self.ref_anchor_head(output).sigmoid()) # nq, bs, 2
            query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / reference_points[..., 2]).unsqueeze(-1)
            query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / reference_points[..., 3]).unsqueeze(-1)

        # main process
        # import ipdb; ipdb.set_trace()
        dropflag = False
        if self.dec_layer_dropout_prob is not None:
            prob = random.random()
            if prob < self.dec_layer_dropout_prob[layer_id]:
                dropflag = True
        if not dropflag:
            output = layer(
                tgt = output,
                tgt_query_pos = query_pos,
                tgt_query_sine_embed = query_sine_embed,
                tgt_key_padding_mask = tgt_key_padding_mask,
                tgt_reference_points = reference_points_input,

                memory = memory,
                memory_key_padding_mask = memory_key_padding_mask,
                memory_level_start_index = level_start_index,
                memory_spatial_shapes = spatial_shapes,
                memory_pos = pos,

                self_attn_mask = tgt_mask,
                cross_attn_mask = memory_mask,
                other_input=other_input
            )
            if isinstance(output, (list, tuple)):
                output, track_res_layer = output
            else:
                track_res_layer = None

        # iter update
        if self.bbox_embed is not None:

            reference_before_sigmoid = inverse_sigmoid(reference_points)
            delta_unsig = self.bbox_embed[layer_id](output)
            outputs_unsig = delta_unsig + reference_before_sigmoid
            new_reference_points = scale_sigmoid(outputs_unsig.sigmoid())

            # select # ref points
            if self.dec_layer_number is not None and layer_id != self.num_layers - 1:
                # import ipdb; ipdb.set_trace()
                nq_now = new_reference_points.shape[0]
                select_number = self.dec_layer_number[layer_id + 1]
                if nq_now != select_number:
                    class_unselected = self.class_embed[layer_id](output) # nq, bs, 91
                    topk_proposals = torch.topk(class_unselected.max(-1)[0], select_number, dim=0)[1] # new_nq, bs
                    new_reference_points = torch.gather(new_reference_points, 0, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid

            if self.rm_detach and 'dec' in self.rm_detach:
                reference_points = new_reference_points
            else:
                reference_points = new_reference_points.detach()
            if self.use_detached_boxes_dec_out:
                ref_points.append(reference_points)
            else:
                ref_points.append(new_reference_points)


        intermediate.append(self.norm(output))
        if self.dec_layer_number is not None and layer_id != self.num_layers - 1:
            if nq_now != select_number:
                output = torch.gather(output, 0, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)) # unsigmoid
        
        return output, reference_points, track_res_layer
    
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
        ref_points = [reference_points]  

        track_res = {}

        for layer_id, layer in enumerate(self.layers):
            # preprocess ref points
            output, reference_points, track_res_layer = self.forward_one_layer(layer_id=layer_id, layer=layer, output=output, memory=memory,reference_points=reference_points, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, level_start_index=level_start_index,spatial_shapes=spatial_shapes,valid_ratios=valid_ratios,pos=pos, tgt_mask=tgt_mask, memory_mask=memory_mask, ref_points=ref_points, intermediate=intermediate)
            track_res[layer_id] = track_res_layer

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
            track_res
        ]


class MotDeformableTransformerDecoderLayer(DeformableTransformerDecoderLayer):
    def __init__(self, args, layer_id, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_levels=4, n_heads=8, n_points=4, use_deformable_box_attn=False, box_attn_type='roi_align', key_aware_type=None, decoder_sa_type='ca', module_seq=...):

        self.args = args
        self.layer_id = layer_id
        self.dropout_p = dropout
        self.n_heads = n_heads

        super(DeformableTransformerDecoderLayer, self).__init__()
        self.module_seq = module_seq
        assert sorted(module_seq) == ['ca', 'ffn', 'sa']

        # cross attention
        # self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        if use_deformable_box_attn:
            self.cross_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points, used_func=box_attn_type)
        else:
            self.cross_attn = self.init_cross_attn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        if decoder_sa_type == 'ca_content':
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
    
    def init_cross_attn(self, d_model, n_levels, n_heads, n_points):
        return MSDeformAttn(d_model, n_levels, n_heads, n_points)
