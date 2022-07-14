#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import pickle
import torch
import torch.utils.data
import numpy as np

from PIL import Image
from torchvision import transforms
from configs.data_config import Relation_Config, NYU40CLASSES

from Method.sun_rgbd.dataset import SUNRGBD

class Total3D_Dataset(SUNRGBD):
    def __init__(self, config, mode):
        super(Total3D_Dataset, self).__init__(config, mode)
        HEIGHT_PATCH = 256
        WIDTH_PATCH = 256
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.mode=='train':
            self.data_transforms = transforms.Compose([
                transforms.Resize((280, 280)),
                transforms.RandomCrop((HEIGHT_PATCH, WIDTH_PATCH)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            self.data_transforms = transforms.Compose([
                transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        self.data_transforms_nocrop = transforms.Compose([
            transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.pil2tensor = transforms.ToTensor()
        self.rel_cfg = Relation_Config()
        self.d_model = int(self.rel_cfg.d_g/4)
        return

    def __getitem__(self, index):

        file_path = self.split[index]
        with open(file_path, 'rb') as f:
            sequence = pickle.load(f)
        image = Image.fromarray(sequence['rgb_img'])
        depth = Image.fromarray(sequence['depth_map'])
        camera = sequence['camera']
        boxes = sequence['boxes']

        # build relational geometric features for each object
        n_objects = boxes['bdb2D_pos'].shape[0]
        # g_feature: n_objects x n_objects x 4
        # Note that g_feature is not symmetric,
        # g_feature[m, n] is the feature of object m contributes to object n.
        # TODO: think about it, do we need to involve the geometric feature from each object itself?
        g_feature = [[((loc2[0] + loc2[2]) / 2. - (loc1[0] + loc1[2]) / 2.) / (loc1[2] - loc1[0]),
                      ((loc2[1] + loc2[3]) / 2. - (loc1[1] + loc1[3]) / 2.) / (loc1[3] - loc1[1]),
                      math.log((loc2[2] - loc2[0]) / (loc1[2] - loc1[0])),
                      math.log((loc2[3] - loc2[1]) / (loc1[3] - loc1[1]))] \
                     for id1, loc1 in enumerate(boxes['bdb2D_pos'])
                     for id2, loc2 in enumerate(boxes['bdb2D_pos'])]

        locs = [num for loc in g_feature for num in loc]

        pe = torch.zeros(len(locs), self.d_model)
        position = torch.from_numpy(np.array(locs)).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        boxes['g_feature'] = pe.view(n_objects * n_objects, self.rel_cfg.d_g)

        # encode class
        cls_codes = torch.zeros([len(boxes['size_cls']), len(NYU40CLASSES)])
        cls_codes[range(len(boxes['size_cls'])), boxes['size_cls']] = 1
        boxes['size_cls'] = cls_codes

        layout = sequence['layout']

        # TODO: If the training error is consistently larger than the test error. We remove the crop and add more intermediate FC layers with no dropout.
        # TODO: Or FC layers with more hidden neurons, which ensures more neurons pass through the dropout layer, or with larger learning rate, longer
        # TODO: decay rate.
        patch = []
        for bdb in boxes['bdb2D_pos']:
            img = image.crop((bdb[0], bdb[1], bdb[2], bdb[3]))
            img = self.data_transforms(img)
            patch.append(img)
        boxes['patch'] = torch.stack(patch)

        image = self.data_transforms_nocrop(image)
        if self.mode != 'test':
            for d, k in zip([camera, layout, boxes], ['world_R_inv', 'lo_inv', 'bdb3d_inv']):
                if k in d.keys():
                    d.pop(k)

        return {'image':image,
                'depth': self.pil2tensor(depth).squeeze(),
                'boxes_batch':boxes,
                'camera':camera,
                'layout':layout,
                'sequence_id': sequence['sequence_id']}

