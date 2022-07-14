#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import torch
import torch.utils.data
import numpy as np

from PIL import Image
from torchvision import transforms
from scipy.spatial import cKDTree

from configs.data_config import pix3d_n_classes

from Method.pix3d.dataset import PIX3D

class MGNet_Dataset(PIX3D):
    def __init__(self, config, mode):
        super(MGNet_Dataset, self).__init__(config, mode)
        self.num_samples_on_each_model = 5000
        self.neighbors = 30

        HEIGHT_PATCH = 256
        WIDTH_PATCH = 256

        self.data_transforms_nocrop = transforms.Compose([
            transforms.Resize((HEIGHT_PATCH, WIDTH_PATCH)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.data_transforms_crop = transforms.Compose([
            transforms.Resize((280, 280)),
            transforms.RandomCrop((HEIGHT_PATCH, WIDTH_PATCH)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return

    def __getitem__(self, index):
        file_path = self.split[index]
        with open(file_path, 'rb') as file:
            sequence = pickle.load(file)

        image = Image.fromarray(sequence['img'])
        class_id = sequence['class_id']
        gt_points = sequence['gt_3dpoints']

        data_transforms = self.data_transforms_crop if self.mode=='train' else self.data_transforms_nocrop

        cls_codes = torch.zeros(pix3d_n_classes)
        cls_codes[class_id] = 1

        tree = cKDTree(gt_points)
        dists, indices = tree.query(gt_points, k=self.neighbors)
        densities = np.array([max(dists[point_set, 1]) ** 2 for point_set in indices])

        if self.mode == 'train':
            p_ids = np.random.choice(gt_points.shape[0], self.num_samples_on_each_model, replace=False)
            gt_points = gt_points[p_ids, :]
            densities = densities[p_ids]

        sample = {'sequence_id':sequence['sample_id'],
                  'img':data_transforms(image),
                  'cls':cls_codes,
                  'mesh_points':gt_points,
                  'densities': densities}

        return sample

