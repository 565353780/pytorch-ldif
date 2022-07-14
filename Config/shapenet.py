#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

class ShapeNetConfig(object):
    def __init__(self):
        self.root_path = "/home/chli/scan2cad/shapenet/"
        self.train_split = self.root_path + "train.json"
        self.test_split = self.root_path + "test.json"
        self.metadata_path = self.root_path
        self.metadata_file = self.metadata_path + "shapenet.json"
        self.mesh_folder = self.metadata_path + "ShapeNetCore.v2/"
        self.classnames = [class_name for class_name in os.listdir(self.mesh_folder)
                           if os.path.isdir(self.mesh_folder + class_name)]
        return

