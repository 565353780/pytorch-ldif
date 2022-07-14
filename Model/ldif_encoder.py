#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class LDIFEncoder(nn.Module):
    def __init__(self, config, n_classes):
        super(LDIFEncoder, self).__init__()
        gauss_kernel_num = 10

        self.config = config

        self.bottleneck_size = self.config['model'].get('bottleneck_size', 2048)
        self.config['model']['bottleneck_size'] = self.bottleneck_size
        self.element_count = self.config['model']['element_count']
        self.sym_element_count = self.config['model']['sym_element_count']
        self.effective_element_count = self.element_count + self.sym_element_count
        self.config['model']['effective_element_count'] = self.effective_element_count
        self.implicit_parameter_length = self.config['model']['implicit_parameter_length']
        self.element_embedding_length = 10 + self.implicit_parameter_length
        self.config['model']['analytic_code_len'] = gauss_kernel_num * self.element_count
        self.config['model']['structured_implicit_vector_len'] = \
            self.element_embedding_length * self.element_count

        self.mlp = nn.Sequential(nn.Linear(self.bottleneck_size + n_classes, self.bottleneck_size),
                                 nn.LeakyReLU(0.2, True),
                                 nn.Linear(self.bottleneck_size, self.bottleneck_size),
                                 nn.LeakyReLU(0.2, True),
                                 nn.Linear(self.bottleneck_size, self.element_count * self.element_embedding_length))

        self.apply(weights_init)
        return

    def forward(self, embedding, size_cls):
        embedding = torch.cat([embedding, size_cls], 1)
        structured_implicit_activations = self.mlp(embedding)
        structured_implicit_activations = torch.reshape(
            structured_implicit_activations, [-1, self.element_count, self.element_embedding_length])
        return structured_implicit_activations

