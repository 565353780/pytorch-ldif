#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from Config.ldif import LDIF_CONFIG

from Module.detector import Detector

class Tester(Detector):
    def __init__(self):
        super(Tester, self).__init__()

        self.test_device = torch.device('cpu')

        self.train_image_ldif_list = []
        self.train_sdf_ldif_list = []
        self.val_image_ldif_list = []
        self.val_sdf_ldif_list = []
        return

    def resetLDIFList(self):
        self.train_image_ldif_list = []
        self.train_sdf_ldif_list = []
        self.val_image_ldif_list = []
        self.val_sdf_ldif_list = []
        return True

    def addTrainLDIF(self, image_ldif, sdf_ldif):
        self.train_image_ldif_list.append(image_ldif)
        self.train_sdf_ldif_list.append(sdf_ldif)
        return True

    def addValLDIF(self, image_ldif, sdf_ldif):
        self.val_image_ldif_list.append(image_ldif)
        self.val_sdf_ldif_list.append(sdf_ldif)
        return True

    def computeRetrieval(self):
        print("[INFO][Tester::computeRetrieval]")
        print("\t train_ldif num =", len(self.train_image_ldif_list))
        print("\t val_ldif num =", len(self.val_image_ldif_list))

        print("\t Accuracy =", 0)

        self.resetLDIFList()
        return True

def demo():
    config = LDIF_CONFIG

    tester = Tester()
    tester.initEnv(config, 'test')
    return True

