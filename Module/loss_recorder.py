#!/usr/bin/env python
# -*- coding: utf-8 -*-

class AverageMeter(object):
    def __init__(self):
        self.reset()
        return

    def reset(self):
        self.val = []
        self.avg = 0
        self.sum = 0
        self.count = 0
        return True

    def update(self, val, n=1):
        if not isinstance(val, list):
            self.sum += val * n
            self.count += n
            self.val += [val] * n
        else:
            self.sum += sum(val)
            self.count += len(val)
            self.val += val
        self.avg = self.sum / self.count
        return True

class LossRecorder(object):
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.loss_recorder = {}
        return

    def update_loss(self, loss_dict):
        for key, item in loss_dict.items():
            if key not in self.loss_recorder:
                self.loss_recorder[key] = AverageMeter()
            self.loss_recorder[key].update(item, self.batch_size)
        return True

