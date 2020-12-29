# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import numpy as np
import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from torch_geometric.data import Data
from functools import reduce
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import pickle


def get_link_labels(pos_edge_index, neg_edge_index):
    
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def graph_data_generator(features, geo_samples, indices_ls, num_boxes_ls, target_indices_ls):
    x_feat = []
    edge_index = [[],[]]
    # pos_edge_index = [[],[]]
    # neg_edge_index = [[],[]]
    num_features = 0
    y_gt = []

    # prob = F.softmax(features[0]['pred_logits'], -1)
    # scores, labels = prob[..., :-1].max(-1)
    # # print(scores, labels)
    # threshed = torch.gt(scores, 0.5)
    # print(scores[threshed])
    # print(target_indices_ls)
    for i in range(len(features)):
        num_features += features[i]["feats_last_layer"][0].shape[0]
        start = i*features[i]["feats_last_layer"][0].shape[0]

        for j in range(len(indices_ls[i][0][0])):
            indices_ls[i][0][0][j] += start
            indices_ls[i][0][1][j] += start

        x_feat.append(features[i]["feats_last_layer"][0])

    x_feat = torch.stack(x_feat)
    x_feat = torch.reshape(x_feat, (num_features, 256)).float()

    for i in range(x_feat.shape[0]):
        for j in range(x_feat.shape[0]):
            if i != j:
                utils.append_to_edge_index(edge_index, i, j)

                # utils.append_to_edge_index(edge_index,i,j)
                x1 = (i // 20)
                y1 = (j // 20)
                if i in indices_ls[x1][0][0] and j in indices_ls[y1][0][0]:
                    idx_i = indices_ls[x1][0][0].tolist().index(i)
                    idx_j = indices_ls[y1][0][0].tolist().index(j)
                    if target_indices_ls[x1][0][idx_i] == target_indices_ls[y1][0][idx_j] and target_indices_ls[x1][0][idx_i] != 99:
                        y_gt.extend([1, 1])
                        # utils.append_to_edge_index(pos_edge_index, i, j)
                    else:
                        y_gt.extend([0, 0])
                        # utils.append_to_edge_index(neg_edge_index, i, j)
                else:
                    y_gt.extend([0, 0])
                    # utils.append_to_edge_index(neg_edge_index, i, j)

    # geos = torch.from_numpy(np.asarray(geos)).double()
    
    edge_index = torch.from_numpy(np.asarray(edge_index)).long()    
    # pos_edge_index = torch.from_numpy(np.asarray(pos_edge_index)).long()
    # neg_edge_index = torch.from_numpy(np.asarray(neg_edge_index)).long()
    y_gt = torch.from_numpy(np.asarray(y_gt)).double()
    data = Data(edge_index=edge_index, 
                x=x_feat, 
                y=y_gt)
    return data

def compute_loss_bce(outputs, gt, device):
    # Define Balancing weight
    positive_vals = gt.sum()

    if positive_vals:
        pos_weight = (gt.shape[0] - positive_vals) / positive_vals

    else: # If there are no positives labels, avoid dividing by zero
        pos_weight = torch.Tensor([0]).to(device)

    # Compute Weighted BCE:
    loss = 0
    
    loss += F.binary_cross_entropy_with_logits(outputs.view(-1),
                                                    gt.view(-1),
                                                    pos_weight= pos_weight)
    return loss


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, model_gnn: torch.nn.Module, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    model_gnn.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('graph_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, geo_samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        geo_samples = geo_samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss_dict, indices_ls, num_boxes_ls, target_indices_ls = criterion(outputs, targets)
        graph_sample = graph_data_generator(outputs, geo_samples, indices_ls, num_boxes_ls, target_indices_ls)
        graph_sample = graph_sample.to(device)

        gnn_out = model_gnn(graph_sample.x, graph_sample.edge_index).to(device)

        loss_bce = compute_loss_bce(gnn_out, graph_sample.y, device)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()

        loss_value = loss_bce + loss_value
        

        losses.backward(retain_graph=True)
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(graph_loss=loss_bce)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    counter = 0
    grapher = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss_dict, indices_ls, num_boxes_ls, target_indices_ls = criterion(outputs, targets)
        graph_sample = graph_data_generator(outputs, indices_ls, num_boxes_ls, target_indices_ls)
        graph_sample = graph_sample.to(device)

        # gnn_out = model_gnn(graph_sample.x, graph_sample.edge_index).to(device)

        grapher.append(graph_sample)

        counter += 1
        weight_dict = criterion.weight_dict
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        thresh_dets = coco_evaluator.eval_imgs["bbox"][0][0][0][0]
        print(thresh_dets)
        # detections_identified = []
        # detections_key = {}
        # for det in thresh_dets:
        #     detections_identified.append(thresh_dets["dtkIds"])
        #     detections_key[thresh_dets["image_id"]] = thresh_dets["dtkIds"]


        # for i in range(len(thresh_dets)):
        #     for j in range(len(thresh_dets)):
        #         if i != j:
        #             det_keys1, det_keys2 = list(filter(lambda a: a != 99, thresh_dets[i]['detKey'])), list(filter(lambda a: a != 99, thresh_dets[j]['detKey']))
        #             print(thresh_dets[i]["dtkIds"])
        #             print(thresh_dets["dtkIds"])
        # exit()

    with open('graph_all.pickle', 'wb') as handle:
        pickle.dump(grapher, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    return stats, coco_evaluator
