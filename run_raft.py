import argparse
import logging
import os
import pickle
import sys
import random

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import skimage.io as sio
from PIL import Image
from PIL import ImageFile

import network_run
from torch.utils.data.dataset import Dataset

# from dataset import generate_image_homogeneous_coordinates, KinectAzureRescaledDataset
from networks import network_utils
from torch.utils.data import DataLoader
import numpy as np

sys.path.append('networks/raft/core')
from networks.raft.core.raft import RAFT


def parse_args():
    parser = argparse.ArgumentParser(description='MARS CNN Script')
    parser.add_argument('--checkpoint', action='append',
                        help='Location of the checkpoints to evaluate.')
    parser.add_argument('--train', type=int, default=1,
                        help='If set to nonzero train the network, otherwise will evaluate.')
    parser.add_argument('--save', type=str, default='',
                        help='The path to save the network checkpoints and logs.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--root', type=str, default='/mars/mnt/dgx/FrameNet')
    parser.add_argument('--epoch', type=int, default=0,
                        help='The epoch to resume training from.')
    parser.add_argument('--iter', type=int, default=0,
                        help='The iteration to resume training from.')
    parser.add_argument('--dataset_pickle_file', type=str, default='./data/scannet_depth_completion_split.pkl')
    parser.add_argument('--dataloader_test_workers', type=int, default=16)
    parser.add_argument('--dataloader_train_workers', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1.e-4)
    parser.add_argument('--save_every_n_iteration', type=int, default=1000,
                        help='Save a checkpoint every n iterations (iterations reset on new epoch).')
    parser.add_argument('--save_every_n_epoch', type=int, default=1,
                        help='Save a checkpoint on the first iteration of every n epochs (independent of iteration).')
    parser.add_argument('--enable_multi_gpu', type=int, default=0,
                        help='If nonzero, use all available GPUs.')
    parser.add_argument('--skip_every_n_image_test', type=int, default=40,
                        help='Skip every n image in the test split.')
    parser.add_argument('--skip_every_n_image_train', type=int, default=1,
                        help='Skip every n image in the test split.')
    parser.add_argument('--eval_test_every_n_iterations', type=int, default=1000,
                        help='Evaluate the network on the test set every n iterations when in training.')
    parser.add_argument('--dataset_type', type=str, default='scannet',
                        help='The dataset loader fromat. Closely related to the pickle file (scannet, nyu, azure).')
    parser.add_argument('--max_epochs', type=int, default=10000,
                        help='Maximum number of epochs for training.')
    parser.add_argument('--depth_loss', type=str, default='L1',
                        help='Depth loss function: L1/L2')

    parser.add_argument('--window', type=int, required=True, nargs='+')
    parser.add_argument('--save_flow', type=int, default=0)
    parser.add_argument('--resize', type=int, default=2)
    parser.add_argument('--azure_ba', type=int, default=0)
    parser.add_argument('--flip_flow', type=int, default=1)
    # RAFT
    parser.add_argument('--raft_model', type=str, default='')
    parser.add_argument('--wdecay', type=float, default=0.0001)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    return parser.parse_args()


class RunDepthRAFT(network_run.DefaultImageNetwork):
    def __init__(self, arguments, train_dataloader, test_dataloader, network_class_creator):
        network_run.DefaultImageNetwork.__init__(self, arguments, train_dataloader, test_dataloader,
                                                 network_class_creator=network_class_creator,
                                                 estimates_depth=True)

        self.fc = self.train_dataloader.dataset.fc
        self.cc = self.train_dataloader.dataset.cc

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.args.learning_rate,
                                                             self.args.num_steps + 100,
                                                             pct_start=0.05, cycle_momentum=False,
                                                             anneal_strategy='linear')

    @staticmethod
    def flow2bearing(flow, fc, cc, normalize=True):
        assert len(flow.shape) == 4
        height, width = flow.shape[2:4]
        xx, yy = np.meshgrid(range(width), range(height))
        pixel = torch.zeros_like(flow)
        match = (flow[:, 0, ...] + torch.from_numpy(xx).cuda(), flow[:, 1] + torch.from_numpy(yy).cuda())
        pixel[:, 0] = (match[0] - cc[0]) / fc[0]
        pixel[:, 1] = (match[1] - cc[1]) / fc[1]
        pixel = torch.cat((pixel, torch.ones_like(pixel[:, 0:1])), dim=1)

        if normalize:
            pixel = F.normalize(pixel)
        return pixel

    @staticmethod
    def rot_bearing_mul(rot, bearing):
        # rot: B x 3 x 3, bearing: B x 3 x H x W
        product = torch.bmm(rot, bearing.view(bearing.shape[0], 3, -1))
        return product.view(bearing.shape)

    @staticmethod
    def ls_2view(r, s):
        hessian = (s * s).sum(dim=1, keepdims=True)
        z = -(s * r).sum(dim=1, keepdims=True) / (hessian + 1e-30)
        e = (r * r).sum(dim=1, keepdims=True) - hessian * (z ** 2)

        invalid_mask = (z <= 0.1)
        invalid_mask |= (z >= 10)
        # invalid_mask |= (e > 0.015 ** 2)
        z[invalid_mask] = 0
        e[invalid_mask] = 0
        hessian[invalid_mask] = 0
        return z, e, hessian

    def triangulation(self, bearings_ref_in_other, t_ref_in_other, flows, residual=False):
        rs, ss = self.pre_triangulation(bearings_ref_in_other, t_ref_in_other, flows, concat=False)
        # get output = (z, residual, hessian)
        outputs = [self.ls_2view(*r_s) for r_s in zip(rs, ss)]
        # weighted sum of z with weight hessian
        hessian = sum([output[2] for output in outputs])
        pred_depths = sum([output[0] * output[2] for output in outputs]) / (hessian + 1e-12)

        if residual:
            # hessian*(z* - z)^2 + residual
            error = torch.sqrt(
                sum([output[2] * (pred_depths - output[0]) ** 2 + output[1] for output in outputs]).clamp_min(0))
            sqrt_hessian = torch.sqrt(hessian)
            return pred_depths, (error, sqrt_hessian)
        else:
            return pred_depths

    def pre_triangulation(self, bearings_ref_in_other, t_ref_in_other, flows, concat=True):
        bearings_other = [self.flow2bearing(flow, self.fc, self.cc, normalize=True) for flow in flows]
        ss = [torch.cross(bearings_other[k], bearings_ref_in_other[k], dim=1) for k in
              range(len(bearings_other))]
        rs = [torch.cross(bearings_other[k], t_ref_in_other[:, k, :, None, None].expand_as(bearings_other[k]), dim=1)
              for k in range(len(bearings_other))]

        if concat:
            s = torch.cat(ss, dim=1)
            r = torch.cat(rs, dim=1)
            return r, s
        else:
            return rs, ss

    @staticmethod
    def depth_ls(r, s):
        pred_depths = -(s * r).sum(dim=1, keepdims=True) / (s * s + 1e-12).sum(dim=1, keepdims=True)
        return pred_depths

    def _network_loss(self, input_batch, cnn_outputs):
        assert isinstance(cnn_outputs, list)
        other_outputs = {}

        _, _, height, width = input_batch['image'].shape
        image_size = height * width

        depths_gt = input_batch['depth'].float().cuda(non_blocking=True)
        depth_mask = depths_gt > 0

        residuals = [torch.sum((depths_gt * s + r) ** 2, dim=1, keepdim=True) for (r, s) in cnn_outputs]
        losses = [{'residual': torch.sum(residual[depth_mask]) / image_size} for residual in residuals]
        losses_map = self.weighted_loss(losses)

        r, s = cnn_outputs[-1]
        pred_depth = self.depth_ls(r, s)
        # Compute the ratios.
        valid_preds = pred_depth[depth_mask]
        valid_gt = depths_gt[depth_mask]

        printable_ratios = network_run.GetDepthPrintableRatios(valid_gt, valid_preds)
        other_outputs.update(printable_ratios)

        return losses_map, other_outputs

    def weighted_loss(self, losses):
        losses_sum = losses[0]
        for key in losses_sum:
            losses_sum[key] = 0.0
            weights = 0.0
            n_predictions = len(losses)
            for i in range(n_predictions):
                i_weight = self.args.gamma ** (n_predictions - i - 1)
                weights += i_weight
                losses_sum[key] += i_weight * losses[i][key]
            losses_sum[key] /= weights
        return losses_sum

    def _on_eval_mode(self):
        self.test_mode = True

    def _on_train_mode(self):
        self.test_mode = False

    def get_flow_flip(self, image1, image2, dims):
        img1 = torch.flip(image1, dims=dims)
        img2 = torch.flip(image2, dims=dims)
        _, flow = self.cnn(img1, img2, iters=self.args.iters, test_mode=self.test_mode)
        flow = torch.flip(flow, dims=dims)
        for dim in dims:
            flow[:, 3 - dim, :, :] *= -1
        return flow

    def get_flow(self, image1, image2):
        flow = self.cnn(image1, image2, iters=self.args.iters, test_mode=self.test_mode)
        if not self.test_mode:
            return flow
        else:
            # In test mode, discard the first unused output from RAFT and only output flow[1]
            if self.args.flip_flow:
                # Flip horizontally and vertically, and average with the original result.
                flip_dims = [[2], [3]]
                flows = [flow[1]] + [self.get_flow_flip(image1, image2, dims) for dims in flip_dims]
                return sum(flows) / len(flows)
            else:
                return flow[1]

    def _call_cnn(self, input_batch):
        bearings_ref_in_other, image1, image2, ts = self.load_inputs(input_batch)

        flows = [self.get_flow(image1, image2[:, k, ...]) for k in range(image2.shape[1])]
        if self.test_mode:
            if self.args.save_flow:
                pred_depth, other = self.triangulation(bearings_ref_in_other, ts, flows, residual=True)
                pred_depth = torch.clamp(pred_depth, min=0.1, max=10)
                return pred_depth, other
            pred_depth = self.triangulation(bearings_ref_in_other, ts, flows)
            pred_depth = torch.clamp(pred_depth, min=0.1, max=10)
            return pred_depth
        else:
            # rearrange so that outer dimension is iteration and the inner one is image index.
            flows = list(map(list, zip(*flows)))
            return [self.pre_triangulation(bearings_ref_in_other, ts, flow) for flow in flows]

    def load_inputs(self, input_batch):
        image1, image2 = input_batch['image'].cuda(non_blocking=True), input_batch['image2'].cuda(non_blocking=True)
        bearing_ref = input_batch['homo'].cuda(non_blocking=True)
        rots = input_batch['rots_ref_in_other'].cuda(non_blocking=True)
        ts = input_batch['ts_ref_in_other'].cuda(non_blocking=True)
        bearings_ref_in_other = [self.rot_bearing_mul(rots[:, k, ...], bearing_ref) for k in range(rots.shape[1])]
        return bearings_ref_in_other, image1, image2, ts

    def _create_optimizer(self, cnn, learning_rate):
        return torch.optim.AdamW(cnn.parameters(), lr=learning_rate, weight_decay=self.args.wdecay,
                                 eps=self.args.epsilon)

    def _run_training_iteration(self, sample_batched, epoch, max_epochs, iteration, max_iters):
        self._on_train_mode()
        self.cnn.train()
        self.optimizer.zero_grad()

        # This function will call the CNN and return its outputs
        cnn_outputs = self._call_cnn(sample_batched)
        self._on_network_output(epoch, iteration, sample_batched, cnn_outputs)

        losses_map, other_outputs_map = self._network_loss(sample_batched, cnn_outputs)

        # Losses map is a mapping from a string to loss.
        total_loss = 0.0
        name_to_value_map = {}
        for name, loss in losses_map.items():
            total_loss += loss
            name_to_value_map[name] = round(loss.item(), 4)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cnn.parameters(), self.args.clip)
        self.optimizer.step()
        self.scheduler.step()

        logging.info('Epoch {0}/{1}, Iter {2}/{3}. Total loss: {4:.4f}. Breakdown: {5}'.format(
            epoch, max_epochs, iteration, max_iters, total_loss.item(), name_to_value_map))
        if other_outputs_map is not None and len(other_outputs_map) > 0:
            logging.info('{}'.format(other_outputs_map))

    def _on_network_output(self, epoch, iteration, input_batch, network_output):
        if self.args.save_flow:
            depth, other = network_output
            residual, sqrt_hessian = other
            output = torch.cat((depth, residual, sqrt_hessian), dim=1).cpu().numpy()

            indexes, scenes = input_batch['frame_index'], input_batch['scene_path']
            for i in range(output.shape[0]):
                save_path = os.path.join(scenes[i], 'flow_depth')
                os.makedirs(save_path, exist_ok=True)
                pickle.dump(output[i, ...], open(os.path.join(save_path, 'flow_depth-%06d.pkl' % indexes[i]), 'wb'))

    @staticmethod
    def depth_projection(rot, t, depth, homo, fc, cc):
        p1 = depth * RunDepthRAFT.rot_bearing_mul(rot, homo)
        # p1 is Bx3xHxW; t is Bx3
        p2 = p1 + t[..., None, None].expand_as(p1)
        projection = p2[:, 0:2, ...] / p2[:, 2:3, ...]
        for i in range(2):
            projection[:, i, ...] = projection[:, i, ...] * fc[i] + cc[i]
        return projection

    @staticmethod
    def save_projection(image1, image2, projection, mask, path):
        h, w, _ = image1.shape
        kp1, kp2, dmatch = [], [], []
        for j in range(50):
            nz = torch.nonzero(mask)
            y, x = nz[random.randrange(len(nz))]
            kp1.append(cv2.KeyPoint(x, y, 1))
            u, v = projection[:, y, x]
            kp2.append(cv2.KeyPoint(u, v, 1))
            dmatch.append(cv2.DMatch(j, j, 1))
        match = cv2.drawMatches(image1, kp1, image2, kp2, dmatch, None)
        print('Save to %s' % path)
        cv2.imwrite(path, cv2.cvtColor(match, cv2.COLOR_RGB2BGR))

    @staticmethod
    def save_projection_flow(image1, image2, flow, projection, mask, path):
        h, w, _ = image1.shape
        kp1, kp2, kp3, dmatch = [], [], [], []
        for j in range(50):
            nz = torch.nonzero(mask)
            y, x = nz[random.randrange(len(nz))]
            kp1.append(cv2.KeyPoint(x, y, 1))
            kp2.append(cv2.KeyPoint(*projection[:, y, x], 1))
            u, v = flow[:, y, x]
            kp3.append(cv2.KeyPoint(x + u, y + v, 1))
            dmatch.append(cv2.DMatch(j, j, 1))
        match = cv2.drawMatches(image1, kp1, image2, kp2, dmatch, None, matchColor=[0, 255, 0])
        cv2.drawMatches(image1, kp1, image2, kp3, dmatch, outImg=match, matchColor=[255, 0, 0],
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
        print('Save to %s' % path)
        cv2.imwrite(path, cv2.cvtColor(match, cv2.COLOR_RGB2BGR))

    @staticmethod
    def save_match(image1, image2, flow, path):
        h, w, _ = image1.shape
        kp1, kp2, dmatch = [], [], []
        for j in range(50):
            x, y = random.randrange(w), random.randrange(h)
            kp1.append(cv2.KeyPoint(x, y, 1))
            u, v = flow[:, y, x]
            kp2.append(cv2.KeyPoint(x + u, y + v, 1))
            dmatch.append(cv2.DMatch(j, j, 1))
        match = cv2.drawMatches(image1, kp1, image2, kp2, dmatch, None)
        print('Save to %s' % path)
        cv2.imwrite(path, cv2.cvtColor(match, cv2.COLOR_RGB2BGR))


def load_raft(args, file):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(file), strict=False)
    return model


def main():
    args = parse_args()
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    network_utils.ConfigureLogging(args.save)

    # First log all the arguments and the values for the record.
    logging.info('sys.argv = {}'.format(sys.argv))
    logging.info('parsed arguments and their values: {}'.format(vars(args)))
    if args.dataset_type == 'azure':
        train_dataset = AzureFlowDataset(usage='test', dataset_pickle_file=args.dataset_pickle_file,
                                         ba_pose=args.azure_ba,
                                         window=args.window, skip_every_n_image=args.skip_every_n_image_train)

        test_dataset = AzureFlowDataset(usage='test', dataset_pickle_file=args.dataset_pickle_file,
                                        ba_pose=args.azure_ba,
                                        window=args.window, skip_every_n_image=args.skip_every_n_image_test)
    else:
        train_dataset = ScanNetFlowDataset(usage='train', window=args.window, resize=args.resize,
                                           dataset_pickle_file=args.dataset_pickle_file,
                                           skip_every_n_image=args.skip_every_n_image_train)

        test_dataset = ScanNetFlowDataset(usage='test', window=args.window, resize=args.resize,
                                          dataset_pickle_file=args.dataset_pickle_file,
                                          skip_every_n_image=args.skip_every_n_image_test)
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.dataloader_train_workers,
                                  pin_memory=True)
    dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.dataloader_test_workers,
                                 pin_memory=True)
    network = RunDepthRAFT(args, dataloader_train, dataloader_test, network_class_creator=(lambda: RAFT(args)))
    # Check if this is training or testing.
    if args.train != 0:
        logging.info('Training the network.')
        if args.epoch != 0:
            resume_model = os.path.join(args.save,
                                        'model-epoch-{0:05d}-iter-{1:05d}.ckpt'.format(args.epoch, args.iter))
            network.load_network_from_file(resume_model)
        elif args.raft_model != "":
            network.cnn = load_raft(args, args.raft_model)
        if args.save == '':
            logging.warning('NO CHECKPOINTS WILL BE SAVED! SET --save FLAG TO SAVE TO A DIRECTORY.')
        network.train(starting_epoch=args.epoch, max_epochs=args.max_epochs)
    else:
        assert args.raft_model is not None
        network.load_network_from_file(args.raft_model)
        network.evaluate()


if __name__ == '__main__':
    main()
