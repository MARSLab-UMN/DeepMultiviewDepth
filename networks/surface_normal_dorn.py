import os
import torch
import torch.nn as nn
import torchvision.models
import collections
import math


class FullImageEncoder(nn.Module):
    def __init__(self, dataset='kitti'):
        super(FullImageEncoder, self).__init__()
        self.global_pooling = nn.AvgPool2d(8, stride=8, padding=(1, 0))
        self.dropout = nn.Dropout2d(p=0.5)
        self.global_fc = nn.Linear(2048 * 4 * 5, 512)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)
        self.upsample = nn.UpsamplingBilinear2d(size=(30, 40))
        self.dataset = dataset

    def forward(self, x):
        x1 = self.global_pooling(x)

        # print('# x1 size:', x1.size())
        x2 = self.dropout(x1)
        x3 = x2.view(-1, 2048 * 4 * 5)
        x4 = self.relu(self.global_fc(x3))
        # print('# x4 size:', x4.size())
        x4 = x4.view(-1, 512, 1, 1)
        # print('# x4 size:', x4.size())
        x5 = self.conv1(x4)
        out = self.upsample(x5)
        return out


class SceneUnderstandingModuleBN(nn.Module):
    def __init__(self, output_channel=136, dataset='kitti', mode='L2'):
        super(SceneUnderstandingModuleBN, self).__init__()
        self.encoder = FullImageEncoder(dataset=dataset)
        self.aspp1 = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=6, dilation=6),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=12, dilation=12),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=18, dilation=18),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512 * 5, 2048, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(2048, output_channel, 1),
            nn.UpsamplingBilinear2d(size=(240, 320))
        )


    def forward(self, x):
        x1 = self.encoder(x)

        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)

        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.concat_process(x6)
        return out

class ResNet(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(ResNet, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(101)](pretrained=pretrained)

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
        # self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer1[0].conv1 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.layer2 = pretrained_model._modules['layer2']

        self.layer3 = pretrained_model._modules['layer3']
        self.layer3[0].conv2.stride = (1, 1)
        self.layer3[0].downsample[0].stride = (1, 1)

        self.layer4 = pretrained_model._modules['layer4']
        self.layer4[0].conv2.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)

        # clear memory
        del pretrained_model

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4

class SurfaceNormalDORN(nn.Module):
    def __init__(self, output_size=(240, 320), pretrained=True, output_channel=3, training_mode='train_L2_loss'):
        super(SurfaceNormalDORN, self).__init__()

        self.output_size = output_size
        self.feature_extractor = ResNet(pretrained=pretrained)
        self.aspp_module = SceneUnderstandingModuleBN(output_channel=output_channel, mode=training_mode)

    def forward(self, x):
        x1 = self.feature_extractor(x)
        x2 = self.aspp_module(x1)
        return torch.nn.functional.normalize(x2, dim=1)