#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import copy
import pickle
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.transforms import Normalize
from torch.utils.tensorboard import SummaryWriter

import resnet
import wideresnet
import utils


class Trainer:
    def __init__(self, trainset, testset, cfg, seed=42, sample_idx=None, prefix=''):
        self.initSeed(seed)
        self.device = cfg['device']
        self.num_epochs = cfg['train']['num_epochs']
        self.architecture = '-'.join(list(map(str, cfg['architecture'].values())))
        if sample_idx is None:
            sample_idx = torch.arange(len(trainset))
            
        def cfg2name(cfg, seed, sample_num):
            name = map(str, [
                 cfg['train']['num_epochs'],
                 cfg['architecture'],
                 cfg['dataset'],
                 seed,
                 sample_num])
            return '_'.join(name)
        self.log_name = '_'.join([prefix, cfg2name(cfg, seed, len(sample_idx))])

        mean, std = self.getDatasetStatistics(trainset.data, sample_idx)
        mean, std = mean.tolist(), std.tolist()
        num_classes = len(trainset.classes)
        
        arch_cfg = cfg['architecture']
        if arch_cfg['name'] == 'resnet':
            model = resnet.string2model(arch_cfg['name'] + str(arch_cfg['depth']), num_classes=num_classes)
        elif arch_cfg['name'] == 'wideresnet':
            d, w = arch_cfg['depth'], arch_cfg['widen_factor']
            drop_rate = arch_cfg['drop_rate']
            model = wideresnet.WideResNet(d, w, drop_rate, num_classes)
        else:
            raise ValueError('wrong architectire')

        self.pipe = nn.Sequential(OrderedDict([
            ('normalize', Normalize(mean=mean, std=std)),
            ('model', model)
        ])).to(self.device)

        opt_cfg = cfg['train']['optimizer']
        self.optimizer = SGD(self.pipe.parameters(),
                        lr=opt_cfg['lr'], 
                        momentum=opt_cfg['momentum'], 
                        weight_decay=opt_cfg['weight_decay'], 
                        nesterov=opt_cfg['nesterov'])

        sch_cfg = cfg['train']['lr_scheduler']
        if sch_cfg['name'] == 'cosine':
            self.lr_scheduler = CosineAnnealingLR(self.optimizer,
                                        T_max=sch_cfg['T_max'], 
                                        eta_min=sch_cfg['eta_min'])
        elif sch_cfg['name'] == 'multistep':
            self.lr_scheduler = MultiStepLR(self.optimizer,
                                        milestones=sch_cfg['milestones'], 
                                        gamma=sch_cfg['gamma'])
        else:
            raise ValueError('wrong scheduler')

        loader_cfg = cfg['train']['dataloader']
        if cfg['target_num'] is not None:
            sample_idx = utils.repeatSampleidx(sample_idx, cfg['target_num'])
        print(f'apparent sample num: {len(sample_idx)}')
        self.trainloader = DataLoader(Subset(trainset, sample_idx), batch_size=loader_cfg['batchsize'], 
                                shuffle=True, 
                                num_workers=loader_cfg['num_workers'], 
                                pin_memory=loader_cfg['pin_memory'])
        self.testloader = DataLoader(testset, batch_size=loader_cfg['batchsize'], 
                                shuffle=False, 
                                num_workers=loader_cfg['num_workers'], 
                                pin_memory=loader_cfg['pin_memory'])


    def initSeed(self, seed):
        #os.environ['PYTHONHASHSEED'] = str(seed)
        #random.seed(seed)
        #np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def train(self):
        self.pipe.train()
        trainloss = 0
        for data in self.trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.pipe(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            self.optimizer.step()
            trainloss += loss.item() * inputs.size()[0]

        trainloss = trainloss / len(self.trainloader.dataset)
        return trainloss


    def test(self):
        self.pipe.eval()
        correct = 0
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.pipe(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / len(self.testloader.dataset)
        return acc


    ## dataset : ndarray(N,H,W,C)
    def getDatasetStatistics(self, dataset, index=None):
        if index is None:
            index = list(range(len(dataset)))
        to_tensor_emu = torch.from_numpy(dataset[index]) / 255
        std, mean = torch.std_mean(to_tensor_emu, dim=(0,1,2), unbiased=False)
        return mean, std


    def run(self):
        writer = SummaryWriter(log_dir=os.path.join('runs', self.log_name))
        with tqdm(range(1, self.num_epochs+1)) as pbar:
            for epoch in pbar:
                pbar.set_description(f'[Epoch {epoch}]')
                trainloss = self.train()
                testacc = self.test()
                writer.add_scalar('Train Loss', trainloss, epoch)
                writer.add_scalar('Test Accuracy', testacc, epoch)
                pbar.set_postfix({'TrainLoss':f'{trainloss:.3f}', 'TestAcc':f'{testacc:.2f}%'})
                self.lr_scheduler.step()


    def save(self, save_dir='output'):
        d = {
            'architecture': self.architecture,
            'mean': self.pipe.normalize.mean,
            'std': self.pipe.normalize.std,
            'state_dict': copy.deepcopy(self.pipe.state_dict()),
        }
        with open(os.path.join(save_dir, self.log_name), 'wb') as f:
            pickle.dump(d, f)


def main():
    from omegaconf import OmegaConf
    from torchvision.datasets import CIFAR10
    parser=argparse.ArgumentParser()
    parser.add_argument('--cfg')
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)
    trainset = CIFAR10(root='./data', train=True, download=False)
    testset = CIFAR10(root='./data', train=False, download=False)
    trainer = Trainer(trainset, testset, cfg)
    print(trainer)
    print(trainer.log_name)

if __name__ == '__main__':
    main()
