'''
Zagoruyko, Sergey and Komodakis, Nikos.
Wide residual networks (2016)
https://arxiv.org/abs/1605.07146
'''

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, drop_rate, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.dropout = nn.Identity() if drop_rate == 0.0 else nn.Dropout(p=drop_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.activation = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride)
            )

    def forward(self, x):
        out = self.dropout(self.conv1(self.activation(self.bn1(x))))
        out = self.conv2(self.activation(self.bn2(out)))

        return out + self.shortcut(x)


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, drop_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'WideResnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        n_stages = [16, 16*k, 32*k, 64*k]

        self.stem = conv3x3(3, n_stages[0])
        self.layer1 = self._make_layer(BasicBlock, n_stages[1], n, drop_rate, stride=1)
        self.layer2 = self._make_layer(BasicBlock, n_stages[2], n, drop_rate, stride=2)
        self.layer3 = self._make_layer(BasicBlock, n_stages[3], n, drop_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(n_stages[3], momentum=0.9)
        self.fc = nn.Linear(n_stages[3], num_classes)
        self.activation = nn.ReLU(inplace=True)
        self.globalpool = nn.AdaptiveAvgPool2d((1,1))

        self._init()


    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0.0) 
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Linear): 
                nn.init.constant_(m.bias, 0.0)


    def _make_layer(self, block, planes, num_blocks, drop_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, drop_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.globalpool(self.activation(self.bn1(out)))
        out = self.fc(out.flatten(1))

        return out


if __name__ == '__main__':
    net = WideResNet(28, 10, 0.3, 10)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
