#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from tqdm import tqdm

from Config.configs import LDIF_CONFIG

from Method.paths import getValidFolderPath, getModelPath

from DataLoader.ldif import LDIF_dataloader
from Model.ldif import LDIF

class Detector(object):
    def __init__(self):
        self.config = {}
        self.device = None
        self.state_dict = None
        self.model = None
        return

    def loadConfig(self, config):
        self.config = config

        log_dict = self.config['log']
        if 'resume_path' in log_dict.keys():
            valid_resume_path = getValidFolderPath(log_dict['resume_path'])
            if valid_resume_path is None:
                print("[ERROR][Detector::loadConfig]")
                print("\t resume_path not exist!")
            else:
                self.config['log']['resume_path'] = valid_resume_path

        if 'path' in log_dict.keys():
            valid_path = getValidFolderPath(log_dict['path'])
            if valid_path is None:
                print("[ERROR][Detector::loadConfig]")
                print("\t path not exist!")
            else:
                self.config['log']['path'] = valid_path
                os.makedirs(valid_path, exist_ok=True)
                name = log_dict['name']
                log_save_path = valid_path + name + "/"
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

    def loadStateDict(self, mode):
        model_path = getModelPath(self.config, mode)
        if model_path is None:
            print("[INFO][Detector::loadStateDict]")
            print("\t trained model not found!")
            return True

        self.state_dict = torch.load(model_path)
        return True

    def loadModel(self, mode):
        self.model = LDIF(self.config, mode)
        self.model.to(self.device)

        if self.state_dict is None:
            return True

        self.model.load_state_dict(self.state_dict['model'])
        return True

    def initEnv(self, config, mode):
        if not self.loadConfig(config):
            print("[ERROR][Detector::initEnv]")
            print("\t loadConfig failed!")
            return False
        if not self.loadDevice():
            print("[ERROR][Detector::initEnv]")
            print("\t loadDevice failed!")
            return False
        if not self.loadStateDict(mode):
            print("[ERROR][Detector::initEnv]")
            print("\t loadStateDict failed!")
            return False
        if not self.loadModel(mode):
            print("[ERROR][Detector::initEnv]")
            print("\t loadModel failed!")
            return False
        return True

    def detect(self, data):
        data = self.to_device(data)
        est_data = self.model(data)
        return est_data

def demo():
    config = LDIF_CONFIG

    detector = Detector()
    detector.initEnv(config, 'test')

    test_dataloader = LDIF_dataloader(config, 'test')
    for data in tqdm(test_dataloader):
        result = detector.detect(data)

        print("==== input ====")
        for key, item in data.items():
            try:
                print(key + ".shape =", item.shape)
            except:
                continue

        print("==== result ====")
        for key, item in result.items():
            try:
                print(key + ".shape =", item.shape)
            except:
                continue
    return True

if __name__ == "__main__":
    demo()

