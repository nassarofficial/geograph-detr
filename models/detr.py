# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d


from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer

import subprocess
import re


import torch_geometric.transforms as T

from torch_geometric.utils import train_test_split_edges, to_undirected
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from torch_geometric.data import DataLoader, Dataset
from torch_cluster import knn_graph
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch_geometric.nn import MessagePassing, GCNConv, PPFConv, PointConv
from torch_geometric.nn.inits import reset
from torch_geometric.nn import global_mean_pool

command = 'nvidia-smi'

class GeoReg(nn.Module):
    """ This is the GeoRegression module that finetuning the pred """
    ## revisit for batching properly
    def __init__(self, out):
        super().__init__()
        self.camera_height = 3
        self.pitch = 0
        self.out = out
        self.linear1 = torch.nn.Linear(256+3+4, 128+3+4)
        self.linear2 = torch.nn.Linear(128+3+4, 128+3+4)
        self.bn1 = torch.nn.BatchNorm1d(128+3+4)
        self.linear3 = torch.nn.Linear(128+3+4, 3)

        # self.conv1 = nn.Conv2d(256+2+4, 128+2+4, 3)
        # self.conv2 = nn.Conv2d(128+2+4, 64+2+4, 3)
        # self.fc1 = nn.Linear(64+2+4,32+2+4)
        # self.fc2 = nn.Linear(32+2+4, 16+2+4)
        # self.fc3 = nn.Linear(16, 2)

    def forward(self, x_feat, pred_geo, cxcywh_bbox):
        pred_x = []
        # print("------: ",len(x_feat), len(pred_geo))
        for i in range(len(x_feat)):
            x = torch.cat((x_feat[i], pred_geo[i]), 0)
            x = torch.cat((x, cxcywh_bbox[i]), 0)
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            x = self.linear3(x)
            pred_x.append(x)
        pred_x = torch.stack(pred_x)
        pred_x = torch.reshape(pred_x, (len(x_feat), 3))
        return pred_x


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss


    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        features_ls, pos_ls, out_ls = [], [], []
        for i in range(samples.tensors.shape[0]):
            out = {}
            if isinstance([samples.tensors[i,:,:,:]], (list, torch.Tensor)):
                samp = nested_tensor_from_tensor_list([samples.tensors[i,:,:,:]])
            features, pos = self.backbone(samp)
            src, mask = features[-1].decompose()

            # print("src.shape: ", src.shape)
            assert mask is not None
            hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

            # print("Out - hs: ", hs.shape)
            outputs_class = self.class_embed(hs)

            # print("Out - outputs_class: ", outputs_class.shape)
            #GNN Goes here.
            outputs_coord = self.bbox_embed(hs).sigmoid()
            # print("Out - outputs_class: ", self.bbox_embed(hs).shape)
            # print("Out - outputs_coord: ", outputs_coord.shape)
            # print("Out - hs: ", hs[-1].shape)

            out['pred_logits'] = outputs_class[-1]
            out['pred_boxes'] = outputs_coord[-1]
            out['feats_last_layer'] = hs[-1]
            
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
            out_ls.append(out)
        return out_ls

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, gnn_model):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.gnn_model = gnn_model

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = {}
        indices_ls, num_boxes_ls, target_indices_ls, target_geos_ls = [], [], [], [] 
        for i in range(len(outputs)):
            outputs_without_aux = {k: v for k, v in outputs[i].items() if k != 'aux_outputs'}
            target_inst = [targets[i]]
            # Retrieve the matching between the outputs of the last layer and the targets
            indices, target_indices, target_geos = self.matcher(outputs_without_aux, target_inst)

            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = sum(len(t["labels"]) for t in target_inst)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs[i].values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
            indices_ls.append(indices) 
            num_boxes_ls.append(num_boxes)
            target_indices_ls.append(target_indices)
            target_geos_ls.append(target_geos)
        # Compute all the requested losses
            for loss in self.losses:
                losses.update(self.get_loss(loss, outputs[i], target_inst, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if 'aux_outputs' in outputs[i]:
                for i, aux_outputs in enumerate(outputs[i]['aux_outputs']):
                    indices, target_indices, target_geos = self.matcher(aux_outputs, target_inst)
                    for loss in self.losses:
                        if loss == 'masks':
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs = {'log': False}
                        l_dict = self.get_loss(loss, aux_outputs, target_inst, indices, num_boxes, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        return losses, indices_ls, num_boxes_ls, target_indices_ls, target_geos_ls


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        results = []
        for i in range(len(outputs)):
            out_logits, out_bbox = outputs[i]['pred_logits'], outputs[i]['pred_boxes']
            assert len(out_logits) == len([target_sizes[i]])
            assert target_sizes.shape[1] == 2

            prob = F.softmax(out_logits, -1)
            scores, labels = prob[..., :-1].max(-1)

            # convert to [x0, y0, x1, y1] format
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            # and from relative [0, 1] to absolute [0, height] coordinates
            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct[:, None, :]

            results.append([{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)][0])

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class EdgePooling(torch.nn.Module):

    def __init__(self, in_channels, dropout=0.3,
                 add_to_edge_score=0.3):
        super(EdgePooling, self).__init__()
        self.in_channels = in_channels
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout

        self.lin = torch.nn.Linear(2 * in_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    @staticmethod
    def compute_edge_score_softmax(raw_edge_score, edge_index, num_nodes):
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)

    @staticmethod
    def compute_edge_score_tanh(raw_edge_score, edge_index, num_nodes):
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(raw_edge_score, edge_index, num_nodes):
        return torch.sigmoid(raw_edge_score)

    def forward(self, x, edge_index):
        e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        e = self.lin(e).view(-1)
        e = F.dropout(e, p=self.dropout, training=self.training)
        e = self.compute_edge_score_sigmoid(e, edge_index, x.size(0))
        e = e + self.add_to_edge_score

        return x, edge_index, e

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='mean') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]

        return self.mlp(tmp)

class GNNNet(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        #1
        # self.conv1 = EdgeConv(256, 64)
        # self.bn1 = BatchNorm1d(64)
        # self.conv6 = EdgeConv(64, out_channels)
        self.edge_pool = EdgePooling(out_channels, add_to_edge_score=0)

        #2
        nn = Seq(Lin(256+3, 64), ReLU(), Lin(64, 64))
        self.conv1 = PointConv(local_nn=nn)

        nn = Seq(Lin(67, 32), ReLU(), Lin(32, 32))
        self.conv2 = PointConv(local_nn=nn)

        self.lin1 = Lin(32, 16)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
            
    def forward(self, x, edge_index, pos):
        # 1
        # x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, training = self.training)
        # x = self.bn1(x)
        # x = self.conv6(x, edge_index)
        # x, edge_index, edge_scores = self.edge_pool(x, edge_index)

        x = F.relu(self.conv1(x, pos, edge_index))
        x = F.relu(self.conv2(x, pos, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x, edge_index, edge_scores = self.edge_pool(x, edge_index)
        # print(edge_scores)
        return edge_scores


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = args.classes

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    matcher = build_matcher(args)

    gnn_model = GNNNet(16).float()

    geo_model = GeoReg(2).float()
    # gnn_model = GAE(GCN(256, 16))

    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}

    weight_dict['loss_giou'] = args.giou_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']

    geo_loss = nn.MSELoss()
    geo_loss.to(device)
    
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses, gnn_model=gnn_model)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, gnn_model, geo_model, criterion, geo_loss, postprocessors
