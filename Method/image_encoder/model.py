#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from Method.resnet import resnet18_full, model_urls
from Method.ldif_encoder.model import LDIFEncoder

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class ImageEncoder(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(ImageEncoder, self).__init__()

        self.encoder = resnet18_full(pretrained=False,
                                     num_classes=num_classes,
                                     input_channels=input_channels)

        self.apply(weights_init)
        self.loadPretrainedResNetEncoder()
        return

    def loadPretrainedResNetEncoder(self):
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = self.encoder.state_dict()
        if pretrained_dict['conv1.weight'].shape != model_dict['conv1.weight'].shape:
            model_dict['conv1.weight'][:,:3,...] = pretrained_dict['conv1.weight']
            pretrained_dict.pop('conv1.weight')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and not k.startswith('fc.')}
        model_dict.update(pretrained_dict)
        self.encoder.load_state_dict(model_dict)
        return True

    def forward(self, image):
        embedding = self.encoder(image)
        return embedding

class ImageLDIFEncoder(nn.Module):
    def __init__(self, config, n_classes):
        super(ImageLDIFEncoder, self).__init__()

        self.config = config

        self.image_encoder = ImageEncoder(self.config['model'].get('bottleneck_size', 2048),
                                          4 if self.config['data'].get('mask', False) else 3)
        self.ldif_encoder = LDIFEncoder(config, n_classes)
        return

    def forward(self, image, size_cls):
        return_dict = {}

        embedding = self.image_encoder.forward(image)
        return_dict['ldif_afeature'] = embedding

        structured_implicit_activations = self.ldif_encoder.forward(embedding, size_cls)
        return_dict['structured_implicit_activations'] = structured_implicit_activations
        return return_dict

