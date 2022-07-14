#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from Method.mgnet.dataset import MGNet_Dataset

def MGNet_collate_fn(batch):
    """
    Data collater.

    Assumes each instance is a dict.
    Applies different collation rules for each field.
    Args:
        batches: List of loaded elements via Dataset.__getitem__
    """
    collated_batch = {}
    # iterate over keys
    for key in batch[0]:
        collated_batch[key] = default_collate([elem[key] for elem in batch])

    return collated_batch

def MGNet_dataloader(config, mode='train'):
    dataloader = DataLoader(dataset=MGNet_Dataset(config, mode),
                            num_workers=config['device']['num_workers'],
                            batch_size=config[mode]['batch_size'],
                            shuffle=(mode == 'train'),
                            collate_fn=MGNet_collate_fn)
    return dataloader

