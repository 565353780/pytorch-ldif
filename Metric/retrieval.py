#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist

class RetrievalMetric(object):
    def __init__(self):
        self.train_image_ldif_list = []
        self.train_sdf_ldif_list = []
        self.val_image_ldif_list = []
        self.val_sdf_ldif_list = []

        self.train_image_ldif_matrix = None
        self.train_sdf_ldif_matrix = None
        self.val_image_ldif_matrix = None
        self.val_sdf_ldif_matrix = None

        self.train_distance_matrix = None
        self.val_distance_matrix = None
        return

    def resetLDIF(self):
        self.train_image_ldif_list = []
        self.train_sdf_ldif_list = []
        self.val_image_ldif_list = []
        self.val_sdf_ldif_list = []

        self.train_image_ldif_matrix = None
        self.train_sdf_ldif_matrix = None
        self.val_image_ldif_matrix = None
        self.val_sdf_ldif_matrix = None

        self.train_distance_matrix = None
        self.val_distance_matrix = None
        return True

    def getLDIFFeature(self, tensor):
        np = tensor.detach().clone().cpu().numpy()
        ldif_np = [ldif.flatten() for ldif in np]
        return ldif_np

    def addTrainLDIF(self, image_ldif, sdf_ldif):
        image_ldif_np = self.getLDIFFeature(image_ldif)
        sdf_ldif_np = self.getLDIFFeature(sdf_ldif)
        self.train_image_ldif_list += image_ldif_np
        self.train_sdf_ldif_list += sdf_ldif_np
        return True

    def addValLDIF(self, image_ldif, sdf_ldif):
        image_ldif_np = self.getLDIFFeature(image_ldif)
        sdf_ldif_np = self.getLDIFFeature(sdf_ldif)
        self.val_image_ldif_list += image_ldif_np
        self.val_sdf_ldif_list += sdf_ldif_np
        return True

    def transToMatrix(self):
        self.train_image_ldif_matrix = np.array(self.train_image_ldif_list)
        self.train_sdf_ldif_matrix = np.array(self.train_sdf_ldif_list)
        self.val_image_ldif_matrix = np.array(self.val_image_ldif_list)
        self.val_sdf_ldif_matrix = np.array(self.val_sdf_ldif_list)
        return True

    def getDistanceMatrix(self):
        self.train_distance_matrix = cdist(self.train_image_ldif_matrix,
                                           self.train_sdf_ldif_matrix,
                                           metric='euclidean')
        self.val_distance_matrix = cdist(self.val_image_ldif_matrix,
                                         self.val_sdf_ldif_matrix,
                                         metric='euclidean')
        return True

    def updateRetrievalMetric(self):
        if not self.transToMatrix():
            print("[ERROR][RetrievalMetric::updateRetrievalMetric]")
            print("\t transToMatrix failed!")
            return False
        if not self.getDistanceMatrix():
            print("[ERROR][RetrievalMetric::updateRetrievalMetric]")
            print("\t getDistanceMatrix failed!")
            return False
        return True

    def getRetrievalMetric(self):
        if not self.updateRetrievalMetric():
            print("[WARN][RetrievalMetric::updateRetrievalMetric]")
            print("\t updateRetrievalMetric failed!")
            self.resetLDIF()
            return None

        metric_dict = {}

        train_diag = np.diag(self.train_distance_matrix)
        val_diag = np.diag(self.val_distance_matrix)

        metric_dict['train_retrieval_dist'] = train_diag.sum()
        metric_dict['val_retrieval_dist'] = val_diag.sum()

        train_retrieval_idx_array = np.argmin(self.train_distance_matrix, axis=1)
        val_retrieval_idx_array = np.argmin(self.val_distance_matrix, axis=1)
        train_match_idx_array = np.array([i for i in range(train_retrieval_idx_array.shape[0])])
        val_match_idx_array = np.array([i for i in range(val_retrieval_idx_array.shape[0])])

        train_match_num = np.sum(train_retrieval_idx_array == train_match_idx_array)
        val_match_num = np.sum(val_retrieval_idx_array == val_match_idx_array)

        metric_dict['train_retrieval_accuracy'] = 1.0 * train_match_num / train_match_idx_array.shape[0]
        metric_dict['val_retrieval_accuracy'] = 1.0 * val_match_num / val_match_idx_array.shape[0]

        train_rank_list = [
            np.sum(self.train_distance_matrix[i] > self.train_distance_matrix[i][i])
            for i in range(self.train_distance_matrix.shape[0])]
        val_rank_list = [
            np.sum(self.val_distance_matrix[i] > self.val_distance_matrix[i][i])
            for i in range(self.val_distance_matrix.shape[0])]

        metric_dict['train_rank_mean'] = np.mean(train_rank_list)
        metric_dict['val_rank_mean'] = np.mean(val_rank_list)

        self.resetLDIF()
        return metric_dict

