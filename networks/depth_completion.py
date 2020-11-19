import logging
from typing import List
import torch
import torch.nn as nn
import torchvision
import collections
import math
from networks.network_utils import *


class ResNetPyramids(nn.Module):
    def __init__(self, in_channels=3, pretrained=False, resnet_arch=18):
        super(ResNetPyramids, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(resnet_arch)](pretrained=pretrained)

        self.channel = in_channels

        self.conv1 = nn.Conv2d(self.channel, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        if pretrained is False:
            weights_init(self.modules(), init_type='kaiming')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


class ChannelReduction(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ChannelReduction, self).__init__()
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.channel_reduction(x)
        return x


class ExtendedUpsample(nn.Module):
    def __init__(self, in_ch, scale_upsample=2, ch_downsample=2, out_spatial=None):
        super(ExtendedUpsample, self).__init__()
        if out_spatial is not None:
            self.extended_upsample = nn.Sequential(
                nn.Conv2d(in_ch, in_ch // ch_downsample, 1),
                nn.BatchNorm2d(in_ch // ch_downsample),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch // ch_downsample, in_ch // ch_downsample, 3, 1, 1),
                nn.BatchNorm2d(in_ch // ch_downsample),
                nn.ReLU(inplace=True),
                nn.Upsample(size=out_spatial, mode='bilinear', align_corners=False),
            )
        else:
            self.extended_upsample = nn.Sequential(
                nn.Conv2d(in_ch, in_ch // ch_downsample, 1),
                nn.BatchNorm2d(in_ch // ch_downsample),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch // ch_downsample, in_ch // ch_downsample, 3, 1, 1),
                nn.BatchNorm2d(in_ch // ch_downsample),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=scale_upsample, mode='bilinear', align_corners=False),
            )

    def forward(self, x):
        x = self.extended_upsample(x)
        return x


class ExtendedFPN(nn.Module):
    def __init__(self, arguments):
        self.args = arguments
        assert (self.args.resnet_arch in [18, 34, 50, 101, 152]), \
            'Only ResNet-18/34/50/101/152 are defined, but got {} layers here!'.format(self.args.resnet_arch)
        super(ExtendedFPN, self).__init__()

        self.resnet_rgb = ResNetPyramids(in_channels=3, pretrained=False, resnet_arch=self.args.resnet_arch)
        self.resnet_normal = ResNetPyramids(in_channels=3, pretrained=False, resnet_arch=self.args.resnet_arch)
        self.resnet_depth = ResNetPyramids(in_channels=1, pretrained=False, resnet_arch=self.args.resnet_arch)

        self.num_encoders = 3

        if self.args.resnet_arch <= 34:
            max_channels = 512
        else:
            max_channels = 2048

        self.input_channels = [max_channels // 8, max_channels // 4, max_channels // 2, max_channels]

        self.feature1_upsampling = nn.Sequential(
            ChannelReduction(in_ch=self.input_channels[0] * self.num_encoders, out_ch=(self.input_channels[0] // 2) * self.num_encoders)
        )

        self.feature2_upsampling = nn.Sequential(
            ChannelReduction(in_ch=self.input_channels[1] * self.num_encoders, out_ch=self.input_channels[0] * self.num_encoders),
            ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=2)
        )

        self.feature3_upsampling = nn.Sequential(
            ChannelReduction(in_ch=self.input_channels[2] * self.num_encoders, out_ch=self.input_channels[1] * self.num_encoders),
            ExtendedUpsample(in_ch=self.input_channels[1] * self.num_encoders, scale_upsample=2, ch_downsample=2),
            ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=2)
        )

        self.feature4_upsampling = nn.Sequential(
            ChannelReduction(in_ch=self.input_channels[3] * self.num_encoders, out_ch=self.input_channels[2] * self.num_encoders),
            ExtendedUpsample(in_ch=self.input_channels[2] * self.num_encoders, out_spatial=(15, 20), ch_downsample=2),
            ExtendedUpsample(in_ch=self.input_channels[1] * self.num_encoders, scale_upsample=2, ch_downsample=2),
            ExtendedUpsample(in_ch=self.input_channels[0] * self.num_encoders, scale_upsample=2, ch_downsample=2)
        )

        if max_channels == 512:
            self.feature_concat = nn.Sequential(
                nn.Conv2d((self.input_channels[0] // 2) * self.num_encoders, 1, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.UpsamplingBilinear2d(scale_factor=4)
            )
        elif max_channels == 2048:
            self.feature_concat = nn.Sequential(
                nn.Conv2d((self.input_channels[0] // 2) * self.num_encoders, (self.input_channels[0] // 4) * self.num_encoders, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d((self.input_channels[0] // 4) * self.num_encoders, 1, 1),
                nn.ReLU(inplace=True),
                nn.UpsamplingBilinear2d(scale_factor=4)
            )

        if self.__class__.__name__ == 'ExtendedFPN':
            logging.info("Backbone: ResNet-{}. Number of parameters in model: {}".format(self.args.resnet_arch,
                                                                                        count_parameters(self)))


    @staticmethod
    def combine_rgbd_features(rgb, normal, depth):
        return torch.cat((rgb, normal, depth), dim=1)


    def forward(self, image, normal, depth_sparse):
        i1, i2, i3, i4 = self.resnet_rgb(image)
        n1, n2, n3, n4 = self.resnet_normal(normal)
        d1, d2, d3, d4 = self.resnet_depth(depth_sparse)

        z1 = self.feature1_upsampling(self.combine_rgbd_features(i1, n1, d1))
        z2 = self.feature2_upsampling(self.combine_rgbd_features(i2, n2, d2))
        z3 = self.feature3_upsampling(self.combine_rgbd_features(i3, n3, d3))
        z4 = self.feature4_upsampling(self.combine_rgbd_features(i4, n4, d4))
        y = self.feature_concat(z1 + z2 + z3 + z4)
        return y
