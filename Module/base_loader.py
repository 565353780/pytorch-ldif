#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch

class BaseLoader(object):
    def __init__(self):
        self.config = {}
        self.device = None
        return

    def loadConfig(self, config):
        self.config = config

        log_dict = self.config['log']
        if 'resume_path' in log_dict.keys():
            resume_path = log_dict['resume_path']
            if resume_path[-1] != "/":
                self.config['log']['resume_path'] += "/"

        if 'path' in log_dict.keys():
            path = log_dict['path']
            if path[-1] != "/":
                self.config['log']['path'] += "/"
            os.makedirs(path, exist_ok=True)
            name = log_dict['name']
            log_save_path = path + name + "/"
            os.makedirs(log_save_path, exist_ok=True)
        return True

    def loadDevice(self):
        device = self.config['device']['device']
        self.device = torch.device(device)
        return True

    def to_device(self, data):
        ndata = {}
        for k, v in data.items():
            if type(v) is torch.Tensor and v.dtype is torch.float32:
                ndata[k] = v.to(self.device)
            else:
                ndata[k] = v
        return ndata

