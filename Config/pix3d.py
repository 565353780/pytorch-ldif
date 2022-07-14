#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

class PIX3DConfig(object):
    def __init__(self):
        self.root_path = "/home/chli/scan2cad/im3d/data/pix3d/"
        self.train_split = self.root_path + "splits/train.json"
        self.test_split = self.root_path + "splits/test.json"
        self.metadata_path = self.root_path + "metadata/"
        self.metadata_file = self.metadata_path + "pix3d.json"
        self.mesh_folder = self.metadata_path + "model/"
        self.classnames = [class_name for class_name in os.listdir(self.mesh_folder)
                           if os.path.isdir(self.mesh_folder + class_name)]
        return

