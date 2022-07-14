#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class LDIFLoss(object):
    def __init__(self, weight=1, config=None):
        self.weight = weight
        self.config = config
        return

    def getLoss(self, est_data, gt_data):
        uniform_sample_loss = nn.MSELoss()(est_data['uniform_class'], gt_data['uniform_class'])
        uniform_sample_loss *= self.config['model']['uniform_loss_weight']

        near_surface_sample_loss = nn.MSELoss()(est_data['near_surface_class'], gt_data['near_surface_class'])
        near_surface_sample_loss *= self.config['model']['near_surface_loss_weight']

        element_centers = est_data['element_centers']

        #  dim = [batch_size, sample_count, 4]
        xyzw_samples = F.pad(element_centers, [0, 1], "constant", 1)

        # dim = [batch_size, sample_count, 3]
        xyzw_samples = torch.matmul(xyzw_samples, gt_data['world2grid'])[..., :3]

        grid = gt_data['grid']
        scale_fac = torch.Tensor(list(grid.shape)[1:]).to(element_centers.device) / 2 - 0.5
        xyzw_samples /= scale_fac

        # dim = [batch_size, 1, 1, sample_count, 3]
        xyzw_samples = xyzw_samples.unsqueeze(1).unsqueeze(1)

        grid = grid.unsqueeze(1)

        gt_sdf_at_centers = F.grid_sample(grid, xyzw_samples, mode='bilinear', padding_mode='zeros')

        gt_sdf_at_centers = torch.where(gt_sdf_at_centers > self.config['data']['coarse_grid_spacing'] / 1.1,
                                        gt_sdf_at_centers, torch.zeros(1).to(gt_sdf_at_centers.device))
        gt_sdf_at_centers *= self.config['model']['lowres_grid_inside_loss_weight']

        element_center_lowres_grid_inside_loss = torch.mean((gt_sdf_at_centers + 1e-04) ** 2) + 1e-05

        bounding_box = self.config['data']['bounding_box']
        lower, upper = -bounding_box, bounding_box
        lower_error = torch.max(lower - element_centers, torch.zeros(1).cuda())
        upper_error = torch.max(element_centers - upper, torch.zeros(1).cuda())
        bounding_box_constraint_error = lower_error * lower_error + upper_error * upper_error
        bounding_box_error = torch.mean(bounding_box_constraint_error)
        inside_box_loss = self.config['model']['inside_box_loss_weight'] * bounding_box_error

        return {'uniform_sample_loss': uniform_sample_loss,
                'near_surface_sample_loss': near_surface_sample_loss,
                'fixed_bounding_box_loss': inside_box_loss,
                'lowres_grid_inside_loss':element_center_lowres_grid_inside_loss}

    def __call__(self, est_data, gt_data):
        loss = self.getLoss(est_data, gt_data)

        if 'sdf_est_data' not in est_data.keys():
            return loss

        sdf_loss = self.getLoss(est_data['sdf_est_data'], gt_data)

        loss['sdf_uniform_sample_loss'] = sdf_loss['uniform_sample_loss']
        loss['sdf_near_surface_sample_loss'] = sdf_loss['near_surface_sample_loss']
        loss['sdf_fixed_bounding_box_loss'] = sdf_loss['fixed_bounding_box_loss']
        loss['sdf_lowres_grid_inside_loss'] = sdf_loss['lowres_grid_inside_loss']
        return loss

