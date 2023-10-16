from motlib.mot_models.network import NETWORK_REGISTRY
from models.dino.backbone import build_backbone
from .layer.deformable_transformer import build_mot_deformable_transformer
from models.dino.dino import PostProcess
from .matcher import build_matcher
from .criterion import build_criterion
from .qim import build_qim
from .model import build_mot_model

import torch
import copy


__all__ = ['build_dinomot']


@NETWORK_REGISTRY.register()
def build_dinomot(args):
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_mot_deformable_transformer(args)

    d_model = transformer.d_model
    hidden_dim = 1024
    query_interaction_layer = build_qim(args.qim_name, args=args, dim_in=d_model, hidden_dim=hidden_dim, dim_out=d_model*2)

    if args.masks:
        raise NotImplementedError
    matcher = build_matcher('MotHungarianMatcher', cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha)

    # prepare weight dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    
    # for DN training
    if args.use_dn:
        weight_dict['loss_ce_dn'] = args.cls_loss_coef
        weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
        weight_dict['loss_giou_dn'] = args.giou_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = build_criterion(args.track_criterion_name, args=args, num_classes=num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses)
    criterion.to(device)

    model = build_mot_model(args.mot_model_name, args, backbone, transformer, criterion, query_interaction_layer)

    postprocessors = {'bbox': PostProcess(num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold)}
    if args.masks:
        raise NotImplementedError

    return model, criterion, postprocessors
