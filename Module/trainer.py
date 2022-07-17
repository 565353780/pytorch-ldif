#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import wandb
import torch
from tqdm import tqdm
from torch.optim import lr_scheduler

from Config.ldif import LDIF_CONFIG

from Method.optimizers import load_optimizer, load_scheduler
from Method.optimizers import load_simple_optimizer

from DataLoader.ldif import LDIF_dataloader

from Metric.retrieval import RetrievalMetric

from Module.loss_recorder import LossRecorder
from Module.detector import Detector

class Trainer(Detector):
    def __init__(self):
        super(Trainer, self).__init__()

        self.train_dataloader = None
        self.test_dataloader = None
        self.optimizer = None
        self.scheduler = None

        self.retrieval_metric = RetrievalMetric()
        return

    def initWandb(self):
        resume = True
        log_dict = self.config['log']

        wandb.init(project=log_dict['project'],
                   config=self.config,
                   dir=log_dict['path'] + log_dict['name'] + "/",
                   name=log_dict['name'],
                   resume=resume)
        wandb.summary['pid'] = os.getpid()
        wandb.summary['ppid'] = os.getppid()
        return True

    def loadDataset(self):
        self.train_dataloader = LDIF_dataloader(self.config, 'train')
        self.test_dataloader = LDIF_dataloader(self.config, 'val')
        return True

    def loadOptimizer(self):
        #  self.optimizer = load_optimizer(self.config, self.model)
        self.optimizer = load_simple_optimizer(self.config, self.model)
        self.scheduler = load_scheduler(self.config, self.optimizer)

        if self.state_dict is not None:
            self.optimizer.load_state_dict(self.state_dict['optimizer'])
            self.scheduler.load_state_dict(self.state_dict['scheduler'])

        wandb.watch(self.model, log=None)
        return True

    def initTrainEnv(self, config):
        if not self.initEnv(config, 'train'):
            print("[ERROR][Trainer::initTrainEnv]")
            print("\t initEnv failed!")
            return False

        if not self.initWandb():
            print("[ERROR][Trainer::initTrainEnv]")
            print("\t initWandb failed!")
            return False
        if not self.loadDataset():
            print("[ERROR][Trainer::initTrainEnv]")
            print("\t loadDevice failed!")
            return False
        if not self.loadOptimizer():
            print("[ERROR][Trainer::initTrainEnv]")
            print("\t loadModel failed!")
            return False
        return True

    def saveModel(self, suffix=None):
        save_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

        log_dict = self.config['log']
        save_folder = log_dict['path'] + log_dict['name'] + "/"

        if not suffix:
            filename = 'model_last.pth'
        else:
            filename = 'model_last.pth'.replace('last', suffix)
        save_path = save_folder + filename
        torch.save(save_dict, save_path)
        return True

    def train_step(self, data):
        self.optimizer.zero_grad()

        data = self.to_device(data)
        est_data = self.model(data)

        self.retrieval_metric.addTrainLDIF(est_data['structured_implicit_activations'],
                                         est_data['sdf_est_data']['structured_implicit_activations'])

        loss = self.model.loss(est_data, data)
        if loss['total'].requires_grad:
            loss['total'].backward()
            self.optimizer.step()

        loss['total'] = loss['total'].item()
        return loss

    def val_step(self, data):
        data = self.to_device(data)
        est_data = self.model(data)

        self.retrieval_metric.addValLDIF(est_data['structured_implicit_activations'],
                                       est_data['sdf_est_data']['structured_implicit_activations'])

        loss = self.model.loss(est_data, data)
        loss['total'] = loss['total'].item()
        return loss

    def outputLr(self):
        lrs = [self.optimizer.param_groups[i]['lr'] for i in range(len(self.optimizer.param_groups))]
        print('[Learning Rate] ' + str(lrs))
        return True

    def outputLoss(self, loss_recorder):
        print("[INFO][Trainer::outputLoss]")
        for loss_name, loss_value in loss_recorder.loss_recorder.items():
            print("\t", loss_name, loss_value.avg)
        return True

    def train_epoch(self, epoch, step):
        batch_size = self.config['train']['batch_size']
        loss_recorder = LossRecorder(batch_size)
        self.model.train(True)

        print_step = self.config['log']['print_step']

        print("[INFO][Trainer::train_epoch]")
        print("\t start train epoch", epoch, "...")

        iter = -1
        for data in tqdm(self.train_dataloader):
            iter += 1
            loss = self.train_step(data)
            loss_recorder.update_loss(loss)

            if (iter % print_step) == 0:
                loss = {f'train_{k}': v for k, v in loss.items()}
                wandb.log(loss, step=step)
                wandb.log({'epoch': epoch}, step=step)
            step += 1
        self.outputLoss(loss_recorder)
        return step

    def val_epoch(self):
        batch_size = self.config['val']['batch_size']
        loss_recorder = LossRecorder(batch_size)
        self.model.train(False)

        print("[INFO][Trainer::val_epoch]")
        print("\t start val epoch ...")

        for data in tqdm(self.test_dataloader):
            loss = self.val_step(data)
            loss_recorder.update_loss(loss)

        self.outputLoss(loss_recorder)
        return loss_recorder.loss_recorder

    def logWandb(self, loss_recorder, epoch, step):
        loss = {f'test_{k}': v.avg for k, v in loss_recorder.items()}
        wandb.log(loss, step=step)
        wandb.log({f'lr{i}': g['lr'] for i, g in enumerate(self.optimizer.param_groups)}, step=step)
        wandb.log({'epoch': epoch + 1}, step=step)
        return True

    def train(self):
        min_eval_loss = 1e8
        epoch = 0
        step = 0

        start_epoch = self.scheduler.last_epoch
        if isinstance(self.scheduler, (lr_scheduler.StepLR, lr_scheduler.MultiStepLR)):
            start_epoch -= 1

        total_epochs = self.config['train']['epochs']
        save_checkpoint = self.config['log']['save_checkpoint']
        for epoch in range(start_epoch, total_epochs):
            print('Epoch (' + str(epoch + 1) + '/' + str(total_epochs) + ').')
            self.outputLr()

            step = self.train_epoch(epoch + 1, step)
            eval_loss_recorder = self.val_epoch()

            retrieval_metric = self.retrieval_metric.getRetrievalMetric()
            print("[INFO][Trainer::train]")
            print("\t RetrievalMetric")
            for key, item in retrieval_metric.items():
                print("\t\t", key, "=", item)
            wandb.log(retrieval_metric)

            eval_loss = eval_loss_recorder['total'].avg
            if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(eval_loss)
            elif isinstance(self.scheduler, (lr_scheduler.StepLR, lr_scheduler.MultiStepLR)):
                self.scheduler.step()
            else:
                print("[ERROR][Trainer::start_train]")
                print("\t scheduler step function not found!")
                return False

            self.logWandb(eval_loss_recorder, epoch, step)

            if save_checkpoint:
                self.saveModel()
            if epoch==-1 or eval_loss < min_eval_loss:
                if save_checkpoint:
                    self.saveModel('best')
                min_eval_loss = eval_loss
                print("[INFO][Trainer::train]")
                print("\t Best VAL Loss")
                for loss_name, loss_value in eval_loss_recorder.items():
                    wandb.summary[f'best_test_{loss_name}'] = loss_value.avg
                    print("\t\t", loss_name, loss_value.avg)
        return True

def demo():
    config = LDIF_CONFIG

    trainer = Trainer()
    trainer.initTrainEnv(config)
    trainer.train()
    return True

