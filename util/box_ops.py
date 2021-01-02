# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area
import math
import numpy as np
def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def get_cartesian(lat,lon):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6371 # radius of the earth
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R *np.sin(lat)
    return x,y,z

def get_lat_lng(x,y,z):
    R = 6371
    lat = np.degrees(np.arcsin(z/R))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon

def streetview_pixel_to_world_coordinates(x, y, yaw, image_width, image_height, o_lat, o_long, height=0):
    camera_height = 3  # ballpark estimate of the number of meters that camera is off the ground
    pitch = 0
    yaw = yaw*math.pi/180
    look_at_angle = x*(2*math.pi)/image_width
    tilt_angle = (image_height/2-y)*math.pi/image_height+pitch
    z = (height-camera_height) / math.tan(min(-1e-2, tilt_angle))
    dx = math.sin(look_at_angle-math.pi+yaw)*z/6371000
    dy = math.cos(look_at_angle-math.pi+yaw)*z/6371000
    lat = o_lat + math.degrees(math.asin(dy))
    lng = o_long + math.degrees(math.asin(dx/math.cos(math.radians(o_lat))))
    return lat, lng

def world_coordinates_to_streetview_pixel(o_lat, o_lng, c_lat, c_lng, height=0):
    camera_height = 3
    pitch = 0

    dx, dy = math.cos(math.radians(c_lat))*math.sin(math.radians(o_lng-c_lng)), math.sin(math.radians(o_lat-c_lat))
    look_at_angle = math.pi + math.atan2(dx, dy) - yaw

    while look_at_angle > 2*math.pi: 
        look_at_angle = look_at_angle-2*math.pi

    while look_at_angle < 0:
        look_at_angle = look_at_angle+2*math.pi

    z = math.sqrt(dx*dx+dy*dy)*6371000
  
    x = (image_width*look_at_angle)/(2*math.pi)
    y = image_height/2 - image_height*(math.atan2(height-camera_height, z)-pitch)/(math.pi) 
    return x, y 

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
