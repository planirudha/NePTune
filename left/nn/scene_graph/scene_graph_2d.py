#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : scene_graph.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 07/19/2018
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os
import collections

import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn

import left.nn.scene_graph.functional_2d as functional_2d


DEBUG = bool(int(os.getenv('DEBUG_SCENE_GRAPH', 0)))

__all__ = ['SceneGraph2D', 'SceneGraphFeature']


class SceneGraphFeature(collections.namedtuple(
    '_SceneGraphFeature', ('scene_feature', 'object_feature', 'relation_feature', 'multi_relation_feature')
)):
    pass


class SceneGraph2D(nn.Module):
    def __init__(self, feature_dim, output_dims, downsample_rate, pool_size=7,
                 incl_relation=True, incl_multi_relation=False, batch=False, norm=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dims = tuple(output_dims)
        self.downsample_rate = downsample_rate
        self.pool_size = pool_size
        self.incl_relation = incl_relation
        self.incl_multi_relation = incl_multi_relation
        self.batch = batch
        self.norm = norm

        self.object_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)
        self.context_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)

        if self.incl_relation:
            self.relation_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)

        if self.incl_multi_relation:
            self.multi_relation_roi_pool = jacnn.PrRoIPool2D(self.pool_size, self.pool_size, 1.0 / downsample_rate)

        if not DEBUG:
            self.context_feature_extract = nn.Conv2d(feature_dim, feature_dim, 1)
            self.object_feature_fuse = nn.Conv2d(feature_dim * 2, output_dims[1], 1)
            self.object_feature_fc = nn.Sequential(nn.ReLU(True), nn.Linear(output_dims[1] * self.pool_size ** 2, output_dims[1]))

            if self.incl_relation:
                self.relation_feature_extract = nn.Conv2d(feature_dim, feature_dim // 2 * 3, 1)
                self.relation_feature_fuse = nn.Conv2d(feature_dim // 2 * 3 + output_dims[1] * 2, output_dims[2], 1)
                self.relation_feature_fc = nn.Sequential(nn.ReLU(True), nn.Linear(output_dims[2] * self.pool_size ** 2, output_dims[2]))

            if self.incl_multi_relation:
                self.multi_relation_feature_extract = nn.Conv2d(feature_dim, feature_dim // 2 * 4, 1)
                self.multi_relation_feature_fuse = nn.Conv2d(
                    feature_dim // 2 * 4 + output_dims[1] * 3, output_dims[3], 1)
                self.multi_relation_feature_fc = nn.Sequential(
                    nn.ReLU(True), nn.Linear(output_dims[3] * self.pool_size ** 2, output_dims[3]))

            self.reset_parameters()
        else:
            def gen_replicate(n):
                def rep(x):
                    return torch.cat(tuple(x for _ in range(n)), dim=1)
                return rep

            self.pool_size = 32
            self.object_roi_pool = jacnn.PrRoIPool2D(32, 32, 1.0 / downsample_rate)
            self.context_roi_pool = jacnn.PrRoIPool2D(32, 32, 1.0 / downsample_rate)
            self.relation_roi_pool = jacnn.PrRoIPool2D(32, 32, 1.0 / downsample_rate)
            self.context_feature_extract = gen_replicate(2)
            self.relation_feature_extract = gen_replicate(3)
            self.object_feature_fuse = jacnn.Identity()
            self.relation_feature_fuse = jacnn.Identity()

            if self.incl_multi_relation:
                self.multi_relation_roi_pool = jacnn.PrRoIPool2D(32, 32, 1.0 / downsample_rate)
                self.multi_relation_feature_extract = gen_replicate(4)
                self.multi_relation_feature_fuse = jacnn.Identity()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(
        self,
        input: torch.Tensor,
        objects: torch.Tensor,
        objects_length: torch.Tensor,
        object_collate=None
    ):
        object_features = input  # Shape: (N, C, H, W)
        context_features = self.context_feature_extract(input)

        relation_features = None
        if self.incl_relation:
            relation_features = self.relation_feature_extract(input)

        if self.incl_multi_relation:
            multi_relation_features = self.multi_relation_feature_extract(input)

        objects_length = objects_length.detach().cpu()
        max_nr_objects = objects_length.max().item()

        if object_collate is None:
            if objects.dim() == 2:
                object_collate = 'concat'
            elif objects.dim() == 3:
                object_collate = 'pad'
            else:
                raise ValueError('Cannot infer object collates from objects.dim().')

        outputs = list()
        objects_index = 0  # for object_collate == 'concat' only
        for i in range(input.size(0)):
            if object_collate == 'concat':
                box = objects[objects_index:objects_index + objects_length[i].item()]
                objects_index += objects_length[i].item()
            else:
                box = objects[i, :objects_length[i].item()]

            num_boxes = box.size(0)
            if self.batch:
                if num_boxes < max_nr_objects:
                    box = torch.cat((
                        box,
                        torch.zeros(max_nr_objects - num_boxes, box.size(1), dtype=box.dtype, device=box.device)
                    ), dim=0)
                num_boxes = box.size(0)  # Update num_boxes after padding

            with torch.no_grad():
                batch_ind = i + torch.zeros(num_boxes, 1, dtype=box.dtype, device=box.device)

                # Generate a "full-image" bounding box
                image_h, image_w = input.size(2) * self.downsample_rate, input.size(3) * self.downsample_rate
                image_box = torch.cat((
                    torch.zeros(num_boxes, 1, dtype=box.dtype, device=box.device),
                    torch.zeros(num_boxes, 1, dtype=box.dtype, device=box.device),
                    image_w + torch.zeros(num_boxes, 1, dtype=box.dtype, device=box.device),
                    image_h + torch.zeros(num_boxes, 1, dtype=box.dtype, device=box.device)
                ), dim=-1)

                box_context_imap = functional_2d.generate_intersection_map(box, image_box, self.pool_size)

                if self.incl_relation:
                    # Meshgrid to obtain the subject and object bounding boxes
                    indices = torch.arange(num_boxes, dtype=torch.int64, device=box.device)
                    sub_id, obj_id = torch.meshgrid(indices, indices, indexing='ij')
                    sub_id = sub_id.contiguous()
                    obj_id = obj_id.contiguous()
                    sub_box = box[sub_id.reshape(-1)]
                    obj_box = box[obj_id.reshape(-1)]

                    # Union box using provided generate_union_box function
                    union_box = functional_2d.generate_union_box(sub_box, obj_box)
                    rel_batch_ind = i + torch.zeros(union_box.size(0), 1, dtype=box.dtype, device=box.device)

                    # Intersection maps
                    sub_union_imap = functional_2d.generate_intersection_map(sub_box, union_box, self.pool_size)
                    obj_union_imap = functional_2d.generate_intersection_map(obj_box, union_box, self.pool_size)

                if self.incl_multi_relation:
                    if num_boxes >= 3:
                        indices = torch.arange(num_boxes, dtype=torch.int64, device=box.device)
                        obj_id1, obj_id2, obj_id3 = torch.meshgrid(indices, indices, indices, indexing='ij')
                        obj_id1 = obj_id1.contiguous()
                        obj_id2 = obj_id2.contiguous()
                        obj_id3 = obj_id3.contiguous()
                        box1 = box[obj_id1.reshape(-1)]
                        box2 = box[obj_id2.reshape(-1)]
                        box3 = box[obj_id3.reshape(-1)]

                        # Compute union boxes iteratively
                        union_box12 = functional_2d.generate_union_box(box1, box2)
                        union_box123 = functional_2d.generate_union_box(union_box12, box3)
                        union_box_multi = union_box123

                        multi_rel_batch_ind = i + torch.zeros(union_box_multi.size(0), 1, dtype=box.dtype, device=box.device)

                        # Intersection maps
                        box1_union_imap = functional_2d.generate_intersection_map(box1, union_box_multi, self.pool_size)
                        box2_union_imap = functional_2d.generate_intersection_map(box2, union_box_multi, self.pool_size)
                        box3_union_imap = functional_2d.generate_intersection_map(box3, union_box_multi, self.pool_size)
                    else:
                        obj_id1 = obj_id2 = obj_id3 = None

            this_context_features = self.context_roi_pool(context_features, torch.cat([batch_ind, image_box], dim=-1))
            x, y = this_context_features.chunk(2, dim=1)
            this_object_features_pooled = self.object_roi_pool(object_features, torch.cat([batch_ind, box], dim=-1))
            this_object_features = self.object_feature_fuse(torch.cat((
                this_object_features_pooled,
                x, y * box_context_imap
            ), dim=1))

            if self.incl_relation:
                this_relation_features_pooled = self.relation_roi_pool(
                    relation_features,
                    torch.cat((rel_batch_ind, union_box), dim=-1)
                )
                x_r, y_r, z_r = this_relation_features_pooled.chunk(3, dim=1)
                sub_feats = this_object_features[sub_id.reshape(-1)]
                obj_feats = this_object_features[obj_id.reshape(-1)]
                this_relation_features = self.relation_feature_fuse(torch.cat((
                    sub_feats, obj_feats,
                    x_r, y_r * sub_union_imap, z_r * obj_union_imap
                ), dim=1))
            else:
                this_relation_features = None

            if self.incl_multi_relation:
                if num_boxes >= 3:
                    this_multi_relation_features_pooled = self.multi_relation_roi_pool(
                        multi_relation_features,
                        torch.cat((multi_rel_batch_ind, union_box_multi), dim=-1)
                    )
                    x_mr, y_mr, z_mr, w_mr = this_multi_relation_features_pooled.chunk(4, dim=1)
                    feats1 = this_object_features[obj_id1.reshape(-1)]
                    feats2 = this_object_features[obj_id2.reshape(-1)]
                    feats3 = this_object_features[obj_id3.reshape(-1)]
                    this_multi_relation_features = self.multi_relation_feature_fuse(torch.cat((
                        feats1, feats2, feats3,
                        x_mr, y_mr * box1_union_imap, z_mr * box2_union_imap, w_mr * box3_union_imap
                    ), dim=1))
                else:
                    this_multi_relation_features = torch.zeros(
                        num_boxes ** 3, self.output_dims[3], device=input.device)
                    obj_id1 = obj_id2 = obj_id3 = None
            else:
                this_multi_relation_features = None

            if not DEBUG:
                this_object_features = self._norm(
                    self.object_feature_fc(this_object_features.view(num_boxes, -1)))
                if self.incl_relation:
                    relation_dim = self.output_dims[2]
                    this_relation_features = self._norm(
                        self.relation_feature_fc(
                            this_relation_features.view(num_boxes ** 2, -1)
                        )
                    ).view(num_boxes, num_boxes, relation_dim)
                if self.incl_multi_relation:
                    multi_relation_dim = self.output_dims[3]
                    this_multi_relation_features = self._norm(
                        self.multi_relation_feature_fc(
                            this_multi_relation_features.view(num_boxes ** 3, -1)
                        )
                    ).view(num_boxes, num_boxes, num_boxes, multi_relation_dim)

            outputs.append(SceneGraphFeature(
                None, this_object_features, this_relation_features, this_multi_relation_features))

        if self.batch:
            return self._batchify(outputs, objects_length, max_nr_objects)
        return outputs

    def _norm(self, x):
        if self.norm and x.numel() > 0:
            return x / x.norm(2, dim=-1, keepdim=True)
        return x

    def _batchify(self, outputs, objects_length, max_nr_objects):
        output1 = torch.stack([x.object_feature for x in outputs], dim=0)
        mask1 = jactorch.length2mask(objects_length, max_nr_objects)
        if self.incl_relation:
            output2 = torch.stack([x.relation_feature for x in outputs], dim=0)
            mask2 = mask1.unsqueeze(2) & mask1.unsqueeze(1)
        else:
            output2 = None
            mask2 = None
        if self.incl_multi_relation:
            output3 = torch.stack([x.multi_relation_feature for x in outputs], dim=0)
            mask3 = mask1.unsqueeze(2).unsqueeze(3) & mask1.unsqueeze(1).unsqueeze(3) & mask1.unsqueeze(1).unsqueeze(2)
        else:
            output3 = None
            mask3 = None
        return (None, (output1, mask1), (output2, mask2), (output3, mask3))
