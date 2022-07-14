#!/usr/bin/env python
# -*- coding: utf-8 -*-

import horovod.torch as hvd
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate

from Method.ldif.dataset import LDIF_Dataset

def LDIF_collate_fn(batch):
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
        try:
            collated_batch[key] = default_collate([elem[key] for elem in batch])
        except TypeError:
            collated_batch[key] = [elem[key] for elem in batch]

    return collated_batch

def LDIF_dataloader(config, mode='train'):
    dataloader = DataLoader(dataset=LDIF_Dataset(config, mode),
                            num_workers=config['device']['num_workers'],
                            batch_size=config[mode]['batch_size'],
                            shuffle=(mode == 'train'),
                            collate_fn=LDIF_collate_fn)
    return dataloader

def HVD_LDIF_dataloader(config, mode='train'):
    dataset = LDIF_Dataset(config, mode)
    sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())

    dataloader = DataLoader(dataset=dataset,
                            num_workers=config['device']['num_workers'],
                            batch_size=config[mode]['batch_size'],
                            collate_fn=LDIF_collate_fn,
                            sampler=sampler)
    return dataloader

