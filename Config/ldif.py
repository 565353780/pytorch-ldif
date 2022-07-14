#!/usr/bin/env python
# -*- coding: utf-8 -*-

DEVICE = {
    'device': 'cuda',
    'num_workers': 8,
}

DATA = {
    'batch_size': 24,
    'dataset': 'pix3d',
    'split': '/home/chli/scan2cad/im3d/data/pix3d/splits/',
    'random_nomask': 0.0,
    'watertight': True,
    'near_surface_samples': 1024,
    'uniform_samples': 1024,
    'bounding_box': 0.7,
    'coarse_grid_spacing': 0.04375,
    'marching_cube_resolution': 256,
}

MODEL = {
    'method': 'LDIF',
    'bottleneck_size': 1536,
    'element_count': 32,
    'sym_element_count': 16,
    'implicit_parameter_length': 32,
    'uniform_loss_weight': 1.0,
    'near_surface_loss_weight': 0.1,
    'lowres_grid_inside_loss_weight': 0.2,
    'inside_box_loss_weight': 10.0,
}

OPTIMIZER = {
    'method': 'Adam',
    'lr': 2e-4,
    'betas': [0.9, 0.999],
    'eps': 1e-08,
    'weight_decay': 0.0,
}

SCHEDULER = {
    'patience': 50,
    'factor': 0.5,
    'threshold': 0.002,
}

TRAIN = {
    'epochs': 40000,
    'phase': 'all',
    'batch_size': 24,
}

VAL = {
    'phase': 'all',
    'batch_size': 24,
}

TEST = {
    'phase': 'all',
    'batch_size': 1,
}

LOG = {
    'project': 'LDIF',
    'name': '20220714_1237',
    'vis_path': 'visualization',
    'save_results': True,
    'vis_step': 100,
    'print_step': 50,
    'save_checkpoint': True,
    'resume_path': './out/ldif/20220714_1237/',
    'path': './out/ldif/',
}

LDIF_CONFIG = {
    'device': DEVICE,
    'data': DATA,
    'model': MODEL,
    'optimizer': OPTIMIZER,
    'scheduler': SCHEDULER,
    'train': TRAIN,
    'val': VAL,
    'test': TEST,
    'log': LOG,
}

