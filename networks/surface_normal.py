import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models
import collections
import math
from networks.warping_2dof_alignment import Warping2DOFAlignment

class ResNetPyramids(nn.Module):
    def __init__(self, in_channels=3):
        super(ResNetPyramids, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(101)](pretrained=True)

        self.channel = in_channels
        self.conv1 = nn.Sequential(collections.OrderedDict([
            ('conv1_1', nn.Conv2d(self.channel, 64, kernel_size=3, stride=2, padding=1, bias=False)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn_2', nn.BatchNorm2d(64)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('conv1_3', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn1_3', nn.BatchNorm2d(128)),
            ('relu1_3', nn.ReLU(inplace=True))
        ]))
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer1[0].conv1 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

    def forward(self, x_input):
        x = self.conv1(x_input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4}

    def freeze(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class SurfaceNormalPrediction(nn.Module):
    def __init__(self, output_size=(240, 320), in_channels=3,
                    training_mode='train_L2_loss',
                    fc_img=np.array([0.5 * 577.87061, 0.5 * 580.25851]),
                    cc_img=np.array([0.5 * 319.87654, 0.5 * 239.87603]),
                    use_mask=False):
        super(SurfaceNormalPrediction, self).__init__()
        self.output_size = output_size
        self.mode = training_mode
        self.use_mask = use_mask

        fc = fc_img
        cc = cc_img
        self.warp_2dof_alignment = Warping2DOFAlignment(fx=fc[0], fy=fc[1], cx=cc[0], cy=cc[1])

        self.resnet_pyramids = ResNetPyramids(in_channels=in_channels)
        self.feature1_upsamping = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature2_upsamping = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature3_upsamping = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 40)),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.feature4_upsamping = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(15, 20)),
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 40)),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.feature_concat = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1),
            nn.UpsamplingBilinear2d(size=(240, 320)),
        )

    def forward(self, x, gravity_tensor, alignment_tensor):
        _, x1 = self.warp_2dof_alignment.warp_with_gravity_center_aligned(x, gravity_tensor, alignment_tensor)
        features = self.resnet_pyramids(x1)
        if self.use_mask:
            feature_mask = x1[:, 0:1] + x1[:, 1:2] + x1[:, 2:3] > 1e-2
            feature_mask = feature_mask.float().detach()
            feature1_mask = nn.functional.interpolate(feature_mask, size=(60, 80), mode='nearest')
            feature2_mask = nn.functional.interpolate(feature_mask, size=(30, 40), mode='nearest')
            feature3_mask = nn.functional.interpolate(feature_mask, size=(15, 20), mode='nearest')
            feature4_mask = nn.functional.interpolate(feature_mask, size=(8, 10), mode='nearest')

            z1 = self.feature1_upsamping(features['x1'] * feature1_mask)
            z2 = self.feature2_upsamping(features['x2'] * feature2_mask)
            z3 = self.feature3_upsamping(features['x3'] * feature3_mask)
            z4 = self.feature4_upsamping(features['x4'] * feature4_mask)
            y = self.feature_concat((z1 + z2 + z3 + z4) * feature1_mask)
        else:
            z1 = self.feature1_upsamping(features['x1'])
            z2 = self.feature2_upsamping(features['x2'])
            z3 = self.feature3_upsamping(features['x3'])
            z4 = self.feature4_upsamping(features['x4'])
            y = self.feature_concat(z1 + z2 + z3 + z4)
        _, z = self.warp_2dof_alignment.inverse_warp_normal_image_with_gravity_center_aligned(y, gravity_tensor, alignment_tensor)
        z = torch.nn.functional.normalize(z, dim=1)
        return z
