import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF

from PIL import Image
import numpy as np

import logging
import os
import sys
from datetime import datetime


def weights_init(modules, init_type='xavier'):
    assert init_type == 'xavier' or init_type == 'kaiming'
    m = modules
    if (isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or \
            isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.ConvTranspose2d) or
            isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Linear)):
        if init_type == 'xavier':
            nn.init.xavier_normal_(m.weight)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight)

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList):
        for m in modules:
            weights_init(m, init_type)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ConfigureLogging(save_path):
    if save_path != '':
        filename = 'log_' + sys.argv[0] + datetime.now().strftime('_%Y%m%d_%H%M%S') + '.log'
        os.makedirs(save_path, exist_ok=True)
        full_file = os.path.join(save_path, filename)
        handlers = [logging.StreamHandler(), logging.FileHandler(full_file)]
    else:
        handlers = [logging.StreamHandler()]

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-1.1s%(asctime)s.%(msecs)03d %(filename)s:%(lineno)d] %(message)s",
        datefmt='%m%d %H:%M:%S',
        handlers=handlers)


def SaveNormalsToImage(normals, filename):
    assert(len(normals.shape) == 3)
    assert(normals.shape[0] == 3)
    normals = normals.transpose([1, 2, 0])
    # Save in FrameNet format.
    normals = (1 + normals) * 127.5
    image = Image.fromarray(normals.astype(np.uint8))
    image.save(filename)


def SaveDepthsToImage(depths, filename):
    if len(depths.shape) == 3:
        assert(depths.shape[0] == 1)
        depths = depths.squeeze()
    assert(len(depths.shape) == 2)
    # The PNG format sometimes does not support writing to uint16 image in Pillow package, so
    # we save in uint32 format.
    image = Image.fromarray((depths * 1000).astype(np.uint32))
    image.save(filename)


def SaveMasksToImage(mask, filename):
    if len(mask.shape) == 3:
        assert(mask.shape[0] == 1)
        mask = mask.squeeze()
    assert(len(mask.shape) == 2)
    image = Image.fromarray(mask)
    image.save(filename)


def SaveRgbToImage(rgb, filename):
    if len(rgb.shape) == 4:
        assert rgb.shape[0] == 1
        rgb =rgb.squeeze()
    assert len(rgb.shape) == 3
    np_image = np.transpose(rgb * 255, axes=[1, 2, 0]).astype(np.uint8)
    image = TVF.to_pil_image(np_image, mode='RGB')
    image.save(filename)


def ComputeDepthErrorStatistics(depth_predicted, depth_gt):
    if isinstance(depth_predicted, torch.Tensor):
        depth_predicted = depth_predicted.detach().cpu().numpy()
    else:
        depth_predicted = np.array(depth_predicted)
    if isinstance(depth_gt, torch.Tensor):
        depth_gt = depth_gt.detach().cpu().numpy()
    else:
        depth_gt = np.array(depth_gt)

    depth_mask = depth_gt > 0
    depth_gt = depth_gt[depth_mask]
    depth_predicted = depth_predicted[depth_mask]

    depth_ratio = np.max(depth_gt / depth_predicted, depth_predicted / depth_gt)
    depth_abs_error = (depth_gt - depth_predicted).abs()

    statistics = {
        'MAD': np.mean(depth_abs_error),
        'RMSE': np.sqrt(np.mean(depth_abs_error ** 2)),
        '1.05': 100 * np.sum(depth_ratio < 1.05) / depth_ratio.shape[0],
        '1.10': 100 * np.sum(depth_ratio < 1.10) / depth_ratio.shape[0],
        '1.25': 100 * np.sum(depth_ratio < 1.25) / depth_ratio.shape[0],
        '1.25^2': 100 * np.sum(depth_ratio < 1.25 ** 2) / depth_ratio.shape[0],
        '1.25^3': 100 * np.sum(depth_ratio < 1.25 ** 3) / depth_ratio.shape[0],
    }

    return statistics


def ComputeNormalErrorStatistics(normal_predicted, normal_gt, mask):
    assert isinstance(normal_predicted, torch.Tensor)
    assert isinstance(normal_gt, torch.Tensor)
    assert isinstance(mask, torch.Tensor)

    normal_predicted = F.normalize(normal_predicted)
    normal_gt = F.normalize(normal_gt)

    length_predicted = torch.norm(normal_predicted)
    length_gt = torch.norm(normal_gt)

    mask = mask & (length_predicted > 0.99) & (length_gt > 0.99)

    dot_products = torch.sum(normal_predicted * normal_gt, dim=1)
    dot_products = torch.clamp(dot_products, min=-1.0, max=1.0)
    angle_errors = torch.acos(dot_products) / np.pi * 180

    mask_np = mask[:, 0, :, :].detach().cpu().numpy() > 0
    angles_np = angle_errors.detach().cpu().numpy()
    normal_errors = angles_np[mask_np]


    statistics = {
        'Mean': np.average(normal_errors),
        'Median': np.median(normal_errors),
        'RMSE': np.sqrt(np.mean(normal_errors ** 2)),
        '5deg': 100 * np.sum(normal_errors < 5.0) / normal_errors.shape[0],
        '7.5deg': 100 * np.sum(normal_errors < 7.5) / normal_errors.shape[0],
        '11.25deg': 100 * np.sum(normal_errors < 11.25) / normal_errors.shape[0],
        '22.5deg': 100 * np.sum(normal_errors < 22.5) / normal_errors.shape[0],
        '30deg': 100 * np.sum(normal_errors < 30) / normal_errors.shape[0],
    }

    return statistics
