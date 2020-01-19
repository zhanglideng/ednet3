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
        self.conv1 = nn.Conv2d(5, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.conv2 = nn.Conv2d(66, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(130, k + 1, kernel_size=5, stride=2, padding=2, bias=False)
        self.relu = nn.PReLU()
        self.d_res_block = D_ResBlock()

        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn65 = nn.BatchNorm2d(65)

    def forward(self, x, a, t):
        # print(x.shape)
        x = torch.cat([x, a], 1)
        x = torch.cat([x, t], 1)
        x = self.relu(self.bn64(self.conv1(x)))
        # print(x.shape)
        t = F.avg_pool2d(t, 2)
        a = F.avg_pool2d(a, 2)
        x = torch.cat([x, a], 1)
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
        a = F.avg_pool2d(a, 2)
        x1 = torch.cat([x1, a], 1)
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


class BottleneckDecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckDecoderBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(in_planes + 32)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(in_planes + 2 * 32)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(in_planes + 3 * 32)
        self.relu4 = nn.ReLU(inplace=True)
        self.bn5 = nn.BatchNorm2d(in_planes + 4 * 32)
        self.relu5 = nn.ReLU(inplace=True)
        self.bn6 = nn.BatchNorm2d(in_planes + 5 * 32)
        self.relu6 = nn.ReLU(inplace=True)
        self.bn7 = nn.BatchNorm2d(inter_planes)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_planes + 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_planes + 2 * 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_planes + 3 * 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_planes + 4 * 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_planes + 5 * 32, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv7 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out1 = self.conv1(self.relu1(self.bn1(x)))
        out1 = torch.cat([x, out1], 1)
        out2 = self.conv2(self.relu2(self.bn2(out1)))
        out2 = torch.cat([out1, out2], 1)
        out3 = self.conv3(self.relu3(self.bn3(out2)))
        out3 = torch.cat([out2, out3], 1)
        out4 = self.conv4(self.relu4(self.bn4(out3)))
        out4 = torch.cat([out3, out4], 1)
        out5 = self.conv5(self.relu5(self.bn5(out4)))
        out5 = torch.cat([out4, out5], 1)
        out6 = self.conv6(self.relu6(self.bn6(out5)))
        out = self.conv7(self.relu7(self.bn7(out6)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        # out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, dropRate=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.conv2(x1)
        out = x + x2
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)


class Dense_decoder(nn.Module):
    def __init__(self, out_channel):
        super(Dense_decoder, self).__init__()
        ############# Block5-up  16-16 ##############
        self.dense_block5 = BottleneckDecoderBlock(128 + 384, 64 + 256)
        self.trans_block5 = TransitionBlock(640 + 192, 32 + 128)
        self.residual_block51 = ResidualBlock(128 + 32)
        self.residual_block52 = ResidualBlock(128 + 32)

        ############# Block6-up 32-32   ##############
        self.dense_block6 = BottleneckDecoderBlock(256 + 32, 128)
        self.trans_block6 = TransitionBlock(384 + 32, 64)
        self.residual_block61 = ResidualBlock(64)
        self.residual_block62 = ResidualBlock(64)

        ############# Block7-up 64-64   ##############
        self.dense_block7 = BottleneckDecoderBlock(64, 64)
        self.trans_block7 = TransitionBlock(128, 32)
        self.residual_block71 = ResidualBlock(32)
        self.residual_block72 = ResidualBlock(32)
        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8 = BottleneckDecoderBlock(32, 32)
        self.trans_block8 = TransitionBlock(64, 16)
        self.residual_block81 = ResidualBlock(16)
        self.residual_block82 = ResidualBlock(16)
        self.conv_refin = nn.Conv2d(19, 20, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.refine3 = nn.Conv2d(20 + 4, 20, kernel_size=3, stride=1, padding=1)
        ##
        self.refine4 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)
        self.refine5 = nn.Conv2d(20, 20, kernel_size=7, stride=1, padding=3)
        self.refine6 = nn.Conv2d(20, out_channel, kernel_size=7, stride=1, padding=3)
        ##
        self.upsample = F.upsample
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

    def forward(self, x, x1, x2, x4, activation=None):
        x42 = torch.cat([x4, x2], 1)
        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x5 = self.residual_block51(x5)
        x5 = self.residual_block52(x5)
        x52 = torch.cat([x5, x1], 1)
        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x6 = self.residual_block61(x6)
        x6 = self.residual_block62(x6)
        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x7 = self.residual_block71(x7)
        x7 = self.residual_block72(x7)
        ##  128 X 128
        x8 = self.trans_block8(self.dense_block8(x7))
        x8 = self.residual_block81(x8)
        x8 = self.residual_block82(x8)
        x8 = torch.cat([x8, x], 1)
        # print x8.size()
        x9 = self.relu(self.conv_refin(x8))

        shape_out = x9.data.size()
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)
        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out, mode='bilinear', align_corners=True)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out, mode='bilinear', align_corners=True)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out, mode='bilinear', align_corners=True)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out, mode='bilinear', align_corners=True)
        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)

        dehaze = self.tanh(self.refine3(dehaze))
        dehaze = self.relu(self.refine4(dehaze))
        dehaze = self.relu(self.refine5(dehaze))

        if activation == 'sig':
            dehaze = self.sig(self.refine6(dehaze))
        else:
            dehaze = self.refine6(dehaze)
        return dehaze


class At(nn.Module):
    def __init__(self):
        super(At, self).__init__()

        ############# 256-256  ##############
        haze_class = models.densenet201(pretrained=True)

        self.conv0 = haze_class.features.conv0
        self.norm0 = haze_class.features.norm0
        self.relu0 = haze_class.features.relu0
        self.pool0 = haze_class.features.pool0

        ############# Block1-down 64-64  ##############
        self.dense_block1 = haze_class.features.denseblock1
        self.trans_block1 = haze_class.features.transition1

        ############# Block2-down 32-32  ##############
        self.dense_block2 = haze_class.features.denseblock2
        self.trans_block2 = haze_class.features.transition2

        ############# Block3-down  16-16 ##############
        self.dense_block3 = haze_class.features.denseblock3
        self.trans_block3 = haze_class.features.transition3

        ############# Block4-up  8-8  ##############
        self.dense_block4 = BottleneckBlock(896, 448)  # 896, 256
        self.trans_block4 = TransitionBlock(896 + 448, 256)  # 1152, 128

        self.decoder_A = Dense_decoder(out_channel=3)
        self.decoder_t = Dense_decoder(out_channel=1)
        # self.decoder_J = Dense_decoder(out_channel=3)

        self.refine1 = nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1)
        self.refine2 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)
        self.refine3 = nn.Conv2d(24, 3, kernel_size=3, stride=1, padding=1)

        self.threshold = nn.Threshold(0.1, 0.1)
        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.upsample = F.upsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        ## 256x256
        x0 = self.pool0(self.relu0(self.norm0(self.conv0(x))))

        ## 64 X 64
        x1 = self.dense_block1(x0)
        # print x1.size()
        x1 = self.trans_block1(x1)

        ###  32x32
        x2 = self.trans_block2(self.dense_block2(x1))
        # print  x2.size()

        ### 16 X 16
        x3 = self.trans_block3(self.dense_block3(x2))

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))

        ######################################
        # J = self.decoder_J(x, x1, x2, x4)
        A = self.decoder_A(x, x1, x2, x4)
        t = self.decoder_t(x, x1, x2, x4, activation='sig')

        t = torch.abs((t)) + (10 ** -10)
        # t = t.repeat(1, 3, 1, 1)

        # haze_reconstruct = J * t + A * (1 - t)
        # J_reconstruct = (x - A * (1 - t)) / t

        # return J_reconstruct, J, A, t, haze_reconstruct
        return A, t


class CNN(nn.Module):
    def __init__(self, k):
        super(CNN, self).__init__()
        self.encoder = EnCoder(k)
        self.decoder = DeCoder()
        self.At = At()

    def forward(self, x):
        A, t = self.At(x)
        x = self.encoder(x, a, t)
        x1 = self.decoder(x)
        return x1, x
