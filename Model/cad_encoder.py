#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from Model.ldif_encoder import LDIFEncoder

class ResNetBlock(nn.Module):
    def __init__(self, num_channels, kernel_size=3, stride=1, layer=nn.Conv3d,
                 normalization=nn.BatchNorm3d, activation=nn.ReLU):
        super().__init__()

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

        self.layer = layer
        self.normalization = normalization
        self.activation = activation

        self.weight_block_0 = nn.Sequential(self.normalization(self.num_channels),
                                            self.activation(inplace=True),
                                            self.layer(self.num_channels,
                                                       self.num_channels,
                                                       kernel_size=self.kernel_size,
                                                       stride=self.stride,
                                                       padding=self.padding,
                                                       bias=False))

        self.weight_block_1 = nn.Sequential(self.normalization(self.num_channels),
                                            self.activation(inplace=True),
                                            self.layer(self.num_channels,
                                                       self.num_channels,
                                                       kernel_size=self.kernel_size,
                                                       stride=self.stride,
                                                       padding=self.padding,
                                                       bias=False))

        self.init_weights()
        return

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return True

    def forward(self, x):
        identity = x
        out = self.weight_block_0(x)
        out = self.weight_block_1(out)

        out = identity + out
        return out

class ResNetEncoder(nn.Module):
    def __init__(self, num_input_channels=1, num_features=None, verbose=False):
        super().__init__()

        if num_features is None:
            num_features = [8, 16, 32, 64, 256]

        self.verbose = verbose
        self.num_features = [num_input_channels] + num_features

        self.network = nn.Sequential(
            # 32 x 32 x 32
            nn.Conv3d(self.num_features[0], self.num_features[1], kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[1]),

            # 32 x 32 x 32
            nn.Conv3d(self.num_features[1], self.num_features[1], kernel_size=4, stride=2, padding=1, bias=False),
            ResNetBlock(self.num_features[1]),

            # 16 x 16 x 16
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[1]),
            nn.Conv3d(self.num_features[1], self.num_features[2], kernel_size=4, stride=2, padding=1, bias=False),
            ResNetBlock(self.num_features[2]),

            # 8 x 8 x 8
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[2]),
            nn.Conv3d(self.num_features[2], self.num_features[3], kernel_size=4, stride=2, padding=1, bias=False),
            ResNetBlock(self.num_features[3]),

            # 4 x 4 x 4
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[3]),
            nn.Conv3d(self.num_features[3], self.num_features[4], kernel_size=4, stride=2, padding=1, bias=False),
            ResNetBlock(self.num_features[4]),

            # 2 x 2 x 2
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[4]),
            nn.Conv3d(self.num_features[4], self.num_features[5], kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(self.num_features[5])
        )

        self.init_weights()
        return

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        return True

    def forward(self, x):
        layers = list(self.network.children())
        for depth, layer in enumerate(layers):
            shape_before = x.data[0].size()
            x = layer(x)
            shape_after = x.data[0].size()

            if self.verbose is True:
                print(f"Layer {depth}: {shape_before} --> {shape_after}")
                self.verbose = False
        return x

class CADLDIFEncoder(nn.Module):
    def __init__(self, config, n_classes):
        super(CADLDIFEncoder, self).__init__()

        self.config = config

        self.cad_encoder = ResNetEncoder(1, [48, 96, 192, 384, 1536])

        self.ldif_encoder = LDIFEncoder(config, n_classes)
        return

    def forward(self, grid, size_cls):
        return_dict = {}

        grid = grid.unsqueeze(1)

        embedding = self.cad_encoder.forward(grid)

        embedding = torch.reshape(embedding, (embedding.shape[0], embedding.shape[1]))
        return_dict['ldif_afeature'] = embedding

        structured_implicit_activations = self.ldif_encoder.forward(embedding, size_cls)
        return_dict['structured_implicit_activations'] = structured_implicit_activations
        return return_dict

