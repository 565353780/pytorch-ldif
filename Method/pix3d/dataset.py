#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from torch.utils.data import Dataset

class PIX3D(Dataset):
    def __init__(self, config, mode):
        '''
        initiate PIX3D dataset for data loading
        :param config: config file
        :param mode: train/val/test mode
        '''
        self.config = config
        if mode == 'val':
            mode = 'test'
        self.mode = mode
        split_file = os.path.join(config['data']['split'], mode + '.json')
        with open(split_file) as file:
            self.split = json.load(file)

    def __len__(self):
        return len(self.split)

