#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from Model.image_encoder import ImageLDIFEncoder
from Model.sdf_encoder import SDFLDIFEncoder
from Model.ldif_decoder import LDIFDecoder

from Loss.ldif import LDIFLoss

class ImageLDIF(nn.Module):
    def __init__(self, config, mode):
        super(ImageLDIF, self).__init__()
        self.n_classes = 9

        self.config = config
        self.mode = mode

        self.image_encoder = ImageLDIFEncoder(config, self.n_classes)

        self.ldif_decoder = LDIFDecoder(config)

        self.mesh_reconstruction_loss = LDIFLoss(1, self.config)

        self.set_mode()
        return

    def set_mode(self):
        if self.config[self.mode]['batch_size'] == 1:
            for m in self.modules():
                if m._get_name().find('BatchNorm') != -1:
                    m.eval()
        return True

    def encodeImage(self, image, size_cls):
        encode_dict = self.image_encoder.forward(image, size_cls)
        return encode_dict['structured_implicit_activations']

    def decodeLDIF(self, structured_implicit_activations):
        return_dict = self.ldif_decoder.forward(structured_implicit_activations)
        return return_dict

    def mesh_reconstruction(self, image, size_cls, samples=None):
        return_dict = self.image_encoder.forward(image, size_cls)

        decoder_return_dict = self.ldif_decoder.forward(return_dict['structured_implicit_activations'], samples)

        return_dict.update(decoder_return_dict)
        return return_dict

    def forward(self, data):
        if 'uniform_samples' not in data.keys():
            return self.mesh_reconstruction(data['img'], data['cls'])

        samples = torch.cat([data['near_surface_samples'], data['uniform_samples']], 1)
        est_data = self.mesh_reconstruction(data['img'], data['cls'], samples=samples)
        len_near_surface = data['near_surface_class'].shape[1]
        est_data['near_surface_class'] = est_data['global_decisions'][:, :len_near_surface, ...]
        est_data['uniform_class'] = est_data['global_decisions'][:, len_near_surface:, ...]
        return est_data

    def loss(self, est_data, gt_data):
        loss = self.mesh_reconstruction_loss(est_data, gt_data)
        total_loss = sum(loss.values())
        for key, item in loss.items():
            loss[key] = item.item()
        return {'total':total_loss, **loss}

class LDIF(nn.Module):
    def __init__(self, config, mode):
        super(LDIF, self).__init__()
        self.n_classes = 9

        self.config = config
        self.mode = mode

        self.image_encoder = ImageLDIFEncoder(config, self.n_classes)

        self.sdf_encoder = SDFLDIFEncoder(config, self.n_classes)

        self.ldif_decoder = LDIFDecoder(config)

        self.mesh_reconstruction_loss = LDIFLoss(1, self.config)

        self.set_mode()
        return

    def set_mode(self):
        if self.config[self.mode]['batch_size'] == 1:
            for m in self.modules():
                if m._get_name().find('BatchNorm') != -1:
                    m.eval()
        return True

    def encodeImage(self, image, size_cls):
        encode_dict = self.image_encoder.forward(image, size_cls)
        return encode_dict['structured_implicit_activations']

    def encodeSDF(self, grid, size_cls):
        encode_dict = self.sdf_encoder.forward(grid, size_cls)
        return encode_dict['structured_implicit_activations']

    def decodeLDIF(self, structured_implicit_activations):
        return_dict = self.ldif_decoder.forward(structured_implicit_activations)
        return return_dict

    def mesh_reconstruction(self, image, size_cls, samples=None):
        return_dict = self.image_encoder.forward(image, size_cls)

        decoder_return_dict = self.ldif_decoder.forward(return_dict['structured_implicit_activations'], samples)

        return_dict.update(decoder_return_dict)
        return return_dict

    def forward(self, data):
        if 'uniform_samples' not in data.keys():
            return self.mesh_reconstruction(data['img'], data['cls'])

        samples = torch.cat([data['near_surface_samples'], data['uniform_samples']], 1)
        est_data = self.mesh_reconstruction(data['img'], data['cls'], samples=samples)
        len_near_surface = data['near_surface_class'].shape[1]
        est_data['near_surface_class'] = est_data['global_decisions'][:, :len_near_surface, ...]
        est_data['uniform_class'] = est_data['global_decisions'][:, len_near_surface:, ...]

        sdf_ldif_encoder = self.sdf_encoder.forward(data['grid'], data['cls'])
        sdf_est_data = self.ldif_decoder.forward(sdf_ldif_encoder['structured_implicit_activations'], samples)
        sdf_est_data['near_surface_class'] = sdf_est_data['global_decisions'][:, :len_near_surface, ...]
        sdf_est_data['uniform_class'] = sdf_est_data['global_decisions'][:, len_near_surface:, ...]
        sdf_est_data.update(sdf_ldif_encoder)
        est_data['sdf_est_data'] = sdf_est_data
        return est_data

    def loss(self, est_data, gt_data):
        loss = self.mesh_reconstruction_loss(est_data, gt_data)
        total_loss = sum(loss.values())
        for key, item in loss.items():
            loss[key] = item.item()
        return {'total':total_loss, **loss}

