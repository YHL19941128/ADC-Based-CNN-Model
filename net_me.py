# -*- coding = utf-8 -*-
# 开发人员：写代码的医生Yin-2020
# 开发时间：2021-05-26下午 12:11
# 文件名称：net_Resnet18.py
# 开发工具：PyCharm

import  torch
from    torch import  nn
from    torch.nn import functional as F

class ResBlock(nn.Module):

    def __init__(self, ch_in, ch_out, stride):

        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out
        out = F.relu(out)

        return out


class Netme(nn.Module):

    def __init__(self, num_class):
        super(Netme, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)

        self.pooling = torch.nn.MaxPool2d(2)

        self.blk1 = ResBlock(16, 32, stride=1)
        self.blk2 = ResBlock(32, 64, stride=1)
        self.blk3 = ResBlock(64, 128, stride=1)
        self.blk4 = ResBlock(128, 256, stride=1)

        self.conv2 = nn.Conv2d(256, 512, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(512)

        self.outlayer1 = nn.Linear(12800, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.outlayer2 = nn.Linear(256, num_class)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x= self.pooling(x)
        x = self.blk1(x)
        x= self.pooling(x)
        x = self.blk2(x)
        x= self.pooling(x)
        x = self.blk3(x)
        x= self.pooling(x)
        x = self.blk4(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.outlayer1(x)
        x = self.dropout(x)
        x = self.outlayer2(x)
        return x
