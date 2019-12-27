import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        # 考虑一下bias的设置
        self.conv = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(128)
        # inplace啥意思？
        self.relu = nn.PReLU()

    def forward(self, x):
        output = self.bn(self.conv(x))
        output = self.relu(output)
        output = self.bn(self.conv(output))
        output = output + x
        return output


class D_ResBlock(nn.Module):
    def __init__(self):
        super(D_ResBlock, self).__init__()
        self.res_block = ResBlock()

    def forward(self, x):
        output = self.res_block(x)
        output = self.res_block(output)
        output = self.res_block(output)
        output = output + x
        return output


class EnCoder(nn.Module):
    def __init__(self, k):
        super(EnCoder, self).__init__()
        self.k = k
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(65, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(129, k + 1, kernel_size=5, stride=2, padding=2, bias=False)
        self.relu = nn.PReLU()
        self.d_res_block = D_ResBlock()

        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn65 = nn.BatchNorm2d(65)

    def forward(self, x, t):
        # print(x.shape)
        x = torch.cat([x, t], 1)
        x = self.relu(self.bn64(self.conv1(x)))
        # print(x.shape)
        t = F.avg_pool2d(t, 2)
        x = torch.cat([x, t], 1)
        x = self.relu(self.bn128(self.conv2(x)))
        # print(x.shape)
        x1 = self.d_res_block(x)
        # print(x1.shape)
        x1 = self.d_res_block(x1)
        x1 = self.d_res_block(x1)
        x1 = self.d_res_block(x1)
        x1 = self.d_res_block(x1)
        x1 = x1 + x
        # print(x1.shape)
        t = F.avg_pool2d(t, 2)
        x1 = torch.cat([x1, t], 1)
        x2 = self.bn65(self.conv3(x1))
        # print(x2.shape)
        indices_map = torch.LongTensor([self.k]).cuda()
        indices_feature = torch.LongTensor([i for i in range(self.k)]).cuda()
        # attention map
        attention_map = torch.index_select(x2, 1, indices_map)
        # print(attention_map.shape)
        x2 = torch.index_select(x2, 1, indices_feature)
        # print(x2.shape)
        x2.mul(attention_map)
        # print(x2.shape)
        return x2


class DeCoder(nn.Module):
    def __init__(self):
        super(DeCoder, self).__init__()
        self.relu = nn.PReLU()
        self.d_res_block = D_ResBlock()
        self.deconv1 = nn.ConvTranspose2d(64, 128, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, dilation=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, 5, stride=2, padding=2, dilation=1, output_padding=1)

        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(3)

    def forward(self, x2):
        # 解码器部分
        x2 = self.relu(self.bn128(self.deconv1(x2)))
        # print(x2.shape)
        x3 = self.d_res_block(x2)
        # print(x3.shape)
        x3 = self.d_res_block(x3)
        x3 = self.d_res_block(x3)
        x3 = self.d_res_block(x3)
        x3 = self.d_res_block(x3)
        x3 = x3 + x2
        # print(x3.shape)
        x3 = self.relu(self.bn64(self.deconv2(x3)))
        # print(x3.shape)
        x3 = self.bn3(self.deconv3(x3))
        # print(x3.shape)
        return x3


class CNN(nn.Module):
    def __init__(self, k):
        super(CNN, self).__init__()
        self.encoder = EnCoder(k)
        self.decoder = DeCoder()

    def forward(self, x, t):
        x = self.encoder(x, t)
        x1 = self.decoder(x)
        return x1, x
