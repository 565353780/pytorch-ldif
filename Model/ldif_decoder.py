#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn

from Method.sdf import reconstruction
from Method.weights import weights_init

from Model.sif import StructuredImplicit
from Model.occnet_decoder import OccNetDecoder

class LDIFDecoder(nn.Module):
    def __init__(self, config):
        super(LDIFDecoder, self).__init__()
        gauss_kernel_num = 10

        self.config = config

        self.element_count = self.config['model']['element_count']
        self.sym_element_count = self.config['model']['sym_element_count']
        self.effective_element_count = self.element_count + self.sym_element_count
        self.config['model']['effective_element_count'] = self.effective_element_count
        self.implicit_parameter_length = self.config['model']['implicit_parameter_length']
        self.element_embedding_length = 10 + self.implicit_parameter_length
        self.config['model']['analytic_code_len'] = gauss_kernel_num * self.element_count
        self.config['model']['structured_implicit_vector_len'] = \
            self.element_embedding_length * self.element_count

        self.decoder = OccNetDecoder(f_dim = self.implicit_parameter_length)

        self.apply(weights_init)
        return

    def eval_implicit_parameters(self, implicit_parameters, samples):
        batch_size, element_count, element_embedding_length = list(implicit_parameters.shape)
        sample_count = samples.shape[-2]
        batched_parameters = torch.reshape(implicit_parameters, [batch_size * element_count, element_embedding_length])
        batched_samples = torch.reshape(samples, [batch_size * element_count, sample_count, -1])
        batched_vals = self.decoder(batched_parameters, batched_samples)
        vals = torch.reshape(batched_vals, [batch_size, element_count, sample_count, 1])
        return vals

    def extract_mesh(self, structured_implicit, resolution=64, extent=0.75, num_samples=10000):
        mesh = reconstruction(structured_implicit, resolution,
                              np.array([-extent] * 3), np.array([extent] * 3),
                              num_samples, False)
        return mesh

    def forward(self, structured_implicit_activations, samples=None):
        return_dict = {}

        structured_implicit = StructuredImplicit.from_activation(
            self.config, structured_implicit_activations, self)
        return_dict['structured_implicit'] = structured_implicit.dict()

        if samples is not None:
            global_decisions, local_outputs = structured_implicit.class_at_samples(samples, True)
            return_dict.update({'global_decisions': global_decisions,
                                'element_centers': structured_implicit.centers})
            return return_dict

        resolution =  self.config['data'].get('marching_cube_resolution', 128)
        mesh = self.extract_mesh(structured_implicit, resolution, self.config['data']['bounding_box'])

        return_dict.update({'sdf': mesh[0], 'mat': mesh[1],
                            'element_centers': structured_implicit.centers})
        return return_dict

