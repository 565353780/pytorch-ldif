#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from tqdm import tqdm
from torch.optim import lr_scheduler

from Config.configs import LDIF_CONFIG

from Method.paths import getModelPath
from Method.dataloaders import LDIF_dataloader, HVD_LDIF_dataloader
from Method.models import LDIF
from Method.optimizers import load_optimizer, load_scheduler

from Module.loss_recorder import LossRecorder
from Module.base_loader import BaseLoader

import horovod.torch as hvd
hvd.init()
if hvd.rank() == 0:
    import wandb
torch.cuda.set_device(hvd.local_rank())

class Trainer(BaseLoader):
    def __init__(self):
        super(Trainer, self).__init__()

        self.train_dataloader = None
        self.test_dataloader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        return

    def initWandb(self):
        if hvd.rank() != 0:
            return True

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

    def loadDataset(self, dataloader):
        self.train_dataloader = dataloader(self.config, 'train')
        self.test_dataloader = dataloader(self.config, 'val')
        return True

    def loadModel(self, model):
        self.model = model(self.config, 'train')
        self.optimizer = load_optimizer(self.config, self.model)
        self.scheduler = load_scheduler(self.config, self.optimizer)

        model_path = getModelPath(self.config)
        if model_path is None:
            print("[INFO][Trainer::loadModel]")
            print("\t trained model not found, start training from 0 epoch...")
        else:
            state_dict = torch.load(model_path)
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])

        self.model.to(self.device)
        if hvd.rank() == 0:
            wandb.watch(self.model, log=None)
        return True

    def loadHVD(self):
        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
        self.optimizer = hvd.DistributedOptimizer(self.optimizer, named_parameters=self.model.named_parameters())
        return True

    def initEnv(self, config, dataloader, model):
        if not self.loadConfig(config):
            print("[ERROR][Trainer::initEnv]")
            print("\t loadConfig failed!")
            return False
        if not self.initWandb():
            print("[ERROR][Trainer::initEnv]")
            print("\t initWandb failed!")
            return False
        if not self.loadDevice():
            print("[ERROR][Trainer::initEnv]")
            print("\t loadDevice failed!")
            return False
        if not self.loadDataset(dataloader):
            print("[ERROR][Trainer::initEnv]")
            print("\t loadDevice failed!")
            return False
        if not self.loadModel(model):
            print("[ERROR][Trainer::initEnv]")
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

    def compute_loss(self, data):
        data = self.to_device(data)
        est_data = self.model(data)

        loss = self.model.loss(est_data, data)
        return loss

    def train_step(self, data):
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        if loss['total'].requires_grad:
            loss['total'].backward()
            self.optimizer.step()

        loss['total'] = loss['total'].item()
        return loss

    def val_step(self, data):
        loss = self.compute_loss(data)
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
        print('=' * 10)
        return True

    def train_epoch(self, epoch, step):
        batch_size = self.config['train']['batch_size']
        loss_recorder = LossRecorder(batch_size)
        self.model.train(True)

        print_step = self.config['log']['print_step']

        if hvd.rank() == 0:
            print("[INFO][Trainer::train_epoch]")
            print("\t start train epoch", epoch, "...")

        iter = -1
        loader = self.train_dataloader
        if hvd.rank() == 0:
            loader = tqdm(self.train_dataloader)
        for data in loader:
            iter += 1
            loss = self.train_step(data)
            loss_recorder.update_loss(loss)

            if (iter % print_step) == 0 and hvd.rank() == 0:
                loss = {f'train_{k}': v for k, v in loss.items()}
                if hvd.rank() == 0:
                    wandb.log(loss, step=step)
                    wandb.log({'epoch': epoch}, step=step)
            step += 1
        if hvd.rank() == 0:
            self.outputLoss(loss_recorder)
        return step

    def val_epoch(self):
        batch_size = self.config['val']['batch_size']
        loss_recorder = LossRecorder(batch_size)
        self.model.train(False)

        if hvd.rank() == 0:
            print("[INFO][Trainer::val_epoch]")
            print("\t start val epoch ...")

        loader = self.test_dataloader
        if hvd.rank() == 0:
            loader = tqdm(self.test_dataloader)
        for data in loader:
            loss = self.val_step(data)
            loss_recorder.update_loss(loss)

        if hvd.rank() == 0:
            self.outputLoss(loss_recorder)
        return loss_recorder.loss_recorder

    def logWandb(self, loss_recorder, epoch, step):
        if hvd.rank() != 0:
            return True

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
            if hvd.rank() == 0:
                print('Epoch (' + str(epoch + 1) + '/' + str(total_epochs) + ').')
                self.outputLr()

            step = self.train_epoch(epoch + 1, step)
            eval_loss_recorder = self.val_epoch()

            eval_loss = eval_loss_recorder['total'].avg
            if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(eval_loss)
            elif isinstance(self.scheduler, (lr_scheduler.StepLR, lr_scheduler.MultiStepLR)):
                self.scheduler.step()
            else:
                print("[ERROR][Trainer::start_train]")
                print("\t scheduler step function not found!")
                return False

            if hvd.rank() != 0:
                continue

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
    dataloader = HVD_LDIF_dataloader
    model = LDIF

    trainer = Trainer()
    trainer.initEnv(config, dataloader, model)
    trainer.loadHVD()
    trainer.train()
    return True

if __name__ == "__main__":
    demo()

