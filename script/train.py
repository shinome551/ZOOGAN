#!/usr/bin/env python
# coding: utf-8

import os
import datetime
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda


class generator(nn.Module):
    def __init__(self, input_dim, img_shape):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = img_shape[0]
        self.input_size = img_shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.LeakyReLU(0.2),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x


class discriminator(nn.Module):
    def __init__(self, img_shape, output_dim):
        super(discriminator, self).__init__()
        self.input_dim = img_shape[0]
        self.output_dim = output_dim
        self.input_size = img_shape[1]

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)
        return x


def getDataloader(dataset, batchsize):
    if dataset == 'cifar10':
        trainset = CIFAR10(root='./data', train=True, download=False)
        testset = CIFAR10(root='./data', train=False, download=False)
        img_shape = (3, 32, 32)
    elif dataset == 'mnist':
        trainset = MNIST(root='./data', train=True, download=True)
        testset = MNIST(root='./data', train=False, download=True)
        img_shape = (1, 28, 28)
    else:
        raise ValueError('wrong dataset.')

    transform = Compose([
        ToTensor(),
        #Lambda(lambda x: x.repeat(3,1,1)),
        #Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        Normalize(mean=(0.5), std=(0.5))
    ])

    trainset.transform = transform
    testset.transform = transform

    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, 
        num_workers=2, 
        pin_memory=True,
        drop_last=True
    )
    testloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    return (trainloader, testloader), img_shape


class GAN(object):
    def __init__(self, cfg):
        self.num_epochs = cfg['num_epochs']
        self.batch_size = cfg['batchsize']
        self.z_dim = cfg['z_dim']
        self.lrG = cfg['optim']['lr_G']
        self.lrD = cfg['optim']['lr_D']
        self.beta1 = cfg['optim']['beta1']
        self.beta2 = cfg['optim']['beta2']
        self.dataset = cfg['dataset']
        self.device = cfg['device']

        # load dataset
        dataloader, self.img_shape = getDataloader(self.dataset, self.batch_size)
        self.dataloader = dataloader[0]

        # networks init
        self.G = generator(self.z_dim, self.img_shape)
        self.D = discriminator(self.img_shape, 1)
        self.G_optimizer = Adam(self.G.parameters(), lr=self.lrG, betas=(self.beta1, self.beta2))
        self.D_optimizer = Adam(self.D.parameters(), lr=self.lrD, betas=(self.beta1, self.beta2))

        self.G.to(self.device)
        self.D.to(self.device)
        self.criterion = nn.BCELoss()

        self.sample_z_ = torch.rand((self.batch_size, self.z_dim)).to(self.device)
        

    def initialize_weights(self):
        for m in self.G.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

        for m in self.D.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()


    def display_process(self, hist, samples, image_frame_dim):
        plt.gcf().clear()
            
        #self.fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)
            
        x = range(len(hist['D_loss']))

        y1 = hist['D_loss']
        y2 = hist['G_loss']
        
        ax1 = self.fig.add_subplot(1, 2, 1)

        ax1.plot(x, y1, label='D_loss')
        ax1.plot(x, y2, label='G_loss')

        ax1.set_xlabel('Iter')
        ax1.set_ylabel('Loss')

        ax1.legend(loc=4)
        ax1.grid(True)
        
        for i in range(image_frame_dim*image_frame_dim):
            ax = self.fig.add_subplot(image_frame_dim, image_frame_dim*2, (int(i/image_frame_dim)+1)*image_frame_dim+i+1, xticks=[], yticks=[])
            if samples[i].shape[2]==3:
                ax.imshow(samples[i])
            else:
                ax.imshow(samples[i][:, :, 0], cmap='gray')

        plt.pause(1)


    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.fig = plt.figure(figsize=(8, 4))

        y_real_ = torch.ones(self.batch_size, 1).to(self.device)
        y_fake_ = torch.zeros(self.batch_size, 1).to(self.device)

        self.initialize_weights()

        self.D.train()
        for epoch in range(self.num_epochs):
            self.G.train()
            for iter, (x_, _) in enumerate(self.dataloader):
                z_ = torch.rand((self.batch_size, self.z_dim))
                x_, z_ = x_.to(self.device), z_.to(self.device)

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = self.criterion(D_real, y_real_)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = self.criterion(D_fake, y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)
                D_fake = self.D(G_)
                G_loss = self.criterion(D_fake, y_real_)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()
                
                if ((iter + 1) % 10) == 0:
                    with torch.no_grad():
                        self.G.eval()
                        samples = self.G(self.sample_z_)
                        samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
                        samples = (samples + 1) / 2
                        self.display_process(self.train_hist, samples, 4)

        print("Training finish!")     
        #plt.show()
        plt.savefig('output/result.png')


    def saveImage(self, num_samples):
        date = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        os.makedirs(f'output/{date}/', exist_ok=True)

        self.G.eval()
        batchsize = 100
        iter_num = num_samples // batchsize
        for i in range(iter_num):
            z_ = torch.rand((batchsize, self.z_dim)).to(self.device)
            samples = self.G(z_)
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
            samples = 255 * (samples + 1) / 2
            for j in range(100):
                img_np = samples[j].astype(np.uint8)
                img_np = img_np[:,:,0] if img_np.shape[2] == 1 else img_np
                img = Image.fromarray(img_np)
                idx = i * batchsize + j
                img.save(f'output/{date}/{idx:0>6}.jpg')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cfg')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg)
    print(cfg)
    gan = GAN(cfg)
    gan.train()
    gan.saveImage(100)

    
if __name__ == '__main__':
    main()