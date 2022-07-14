#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset

class SUNRGBD(Dataset):
    def __init__(self, config, mode):
        '''
        initiate SUNRGBD dataset for data loading
        :param config: config file
        :param mode: train/val/test mode
        '''
        self.config = config
        if mode == 'val':
            mode = 'test'
        self.mode = mode
        split_file = os.path.join(config['data']['split'], mode + '.json')
        with open(split_file) as file:
            split = json.load(file)
        self.split = []
        skipped = 0
        for s in tqdm(split):
            if os.path.exists(s):
                self.split.append(s)
            else:
                skipped += 1
        print(f'{skipped}/{len(split)} missing samples')

    def __len__(self):
        return len(self.split)

