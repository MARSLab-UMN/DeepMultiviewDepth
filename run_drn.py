import argparse
import logging
import numpy as np
import os
import sys
import torch
import torch.autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import cv2

# from dataset import *
import network_run
from networks.depth_completion import *
from networks.depth_refinement_network import DRN

from networks.surface_normal import *
from networks.surface_normal_dorn import *
import networks.network_utils as network_utils


def ParseCmdLineArguments():
    parser = argparse.ArgumentParser(description='MARS CNN Script')
    parser.add_argument('--checkpoint', action='append',
                        help='Location of the checkpoints to evaluate.')
    parser.add_argument('--train', type=int, default=1,
                        help='If set to nonzero train the network, otherwise will evaluate.')
    parser.add_argument('--save', type=str, default='',
                        help='The path to save the network checkpoints and logs.')
    parser.add_argument('--save_visualization', type=str, default='',
                        help='Saving network output images.')
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
    parser.add_argument('--resnet_arch', type=int, default=18,
                        help='ResNet architecture for ModifiedFPN (18/34/50/101/152)')
    parser.add_argument('--surface_normal_checkpoint', type=str, default='',
                        help='Surface normal checkpoint path is a required field.')
    parser.add_argument('--predicted_normal_subdirectory', type=str, default='DORN_acos_bs16_inference',
                        help='Predicted surface normal subdir path is a required field.')
    parser.add_argument('--dataset_type', type=str, default='scannet',
                        help='The dataset loader fromat. Closely related to the pickle file (scannet, nyu, azure).')
    parser.add_argument('--max_epochs', type=int, default=10000,
                        help='Maximum number of epochs for training.')
    parser.add_argument('--depth_loss', type=str, default='L1',
                        help='Depth loss function: L1/L2')

    # Depth Refinement Network (DRN)
    parser.add_argument('--predicted_flowdepth_subdirectory', type=str, default='flow_depth_kf_v4',
                        help='Predicted flow depth subdir path is a required field.')
    parser.add_argument('--refinement_iterations', type=int, default=5,
                        help='Iterative refinement iterations')
    parser.add_argument('--uncertainty_threshold', type=float, default=-1.0,
                        help='Depth uncertainty thresholds.')
    return parser.parse_args()



class RunIterativeDepthCompletion(network_run.DefaultImageNetwork):
    def __init__(self, arguments, train_dataloader, test_dataloader, network_class_creator):
        super(RunIterativeDepthCompletion, self).__init__(arguments, train_dataloader, test_dataloader,
                                                 network_class_creator=network_class_creator,
                                                 estimates_depth=True)
        self.output_path = arguments.save_visualization
        if self.output_path != '':
            if not os.path.exists(self.output_path):
                os.mkdir(self.output_path)
        self.save_idx = 0


    def create_color_depth_image(self, depth, thres=5.0):
        valid_mask_depth = depth > 0
        threshold_mask = depth >= thres
        depth[threshold_mask] = thres

        output_color_depth_img = cv2.applyColorMap(np.uint8(depth * 255 / thres), cv2.COLORMAP_JET)
        output_color_depth_img = cv2.cvtColor(output_color_depth_img, cv2.COLOR_RGB2BGR)
        output_color_depth_img[:, :, 0][~valid_mask_depth] = 0
        output_color_depth_img[:, :, 1][~valid_mask_depth] = 0
        output_color_depth_img[:, :, 2][~valid_mask_depth] = 139
        return output_color_depth_img


    def create_color_error_depth_image(self, depth_error, mask, thres=2.0):
        valid_mask_depth = mask > 0
        threshold_mask = depth_error >= thres
        depth_error[threshold_mask] = thres

        output_color_depth_img = cv2.applyColorMap(np.uint8(depth_error * 255 / thres), cv2.COLORMAP_JET)
        output_color_depth_img = cv2.cvtColor(output_color_depth_img, cv2.COLOR_RGB2BGR)
        output_color_depth_img[:, :, 0][~valid_mask_depth] = 128
        output_color_depth_img[:, :, 1][~valid_mask_depth] = 128
        output_color_depth_img[:, :, 2][~valid_mask_depth] = 128
        return output_color_depth_img


    def visualization(self, output_pred, input_batch):
        output_path = self.output_path

        # rgb images
        rgb = input_batch['image'][0].squeeze().detach().cpu().numpy()
        rgb_image = np.transpose(rgb * 255, axes=[1, 2, 0]).astype(np.uint8)

        depth_flow_img = self.create_color_depth_image(
            input_batch['flow_depth'][0, ...].squeeze().detach().cpu().numpy(),
            thres=5.0)

        depth_gt_img = self.create_color_depth_image(input_batch['depth'][0].squeeze().detach().cpu().numpy(),
                                                     thres=5.0)
        depth_pred_img = self.create_color_depth_image(output_pred['d'][-1][0].squeeze().detach().cpu().numpy(),
                                                       thres=5.0)
        depth_error_img = self.create_color_error_depth_image(
            (output_pred['d'][-1][0].cpu()-input_batch['depth'][0]).abs().squeeze().detach().cpu().numpy(),
            (input_batch['depth'][0] > 0).squeeze().detach().cpu().numpy(), thres=2.0)

        depth_uncertainty_img = self.create_color_error_depth_image(
            torch.exp(output_pred['u'][-1][0]).cpu().abs().squeeze().detach().cpu().numpy(),
            torch.ones_like(input_batch['depth'][0]).squeeze().detach().cpu().numpy(), thres=2.0)

        normal_pred = torch.nn.functional.normalize(input_batch['predicted_normal'])
        normal_pred = normal_pred[0].squeeze().detach().cpu().numpy().transpose([1, 2, 0])
        normal_pred = (1 + normal_pred) * 127.5

        output_img = np.concatenate((rgb_image, normal_pred, depth_flow_img,
                                     depth_pred_img, depth_gt_img, depth_uncertainty_img, depth_error_img), axis=1)
        image = Image.fromarray(output_img.astype(np.uint8))
        image.save(os.path.join(output_path, 'viz_{0:06d}.png'.format(self.save_idx)))
        self.save_idx += 1


    def _call_cnn(self, input_batch):
        flow_depth = input_batch['flow_depth'].cuda(non_blocking=True)
        flow_depth_confidence_scores = input_batch['flow_depth_confidence_scores'].cuda(non_blocking=True)
        rgb_image = input_batch['image'].cuda(non_blocking=True)
        predicted_normals = input_batch['predicted_normal'].cuda(non_blocking=True)
        predicted_normals = torch.nn.functional.normalize(predicted_normals)

        depth_complete = self.cnn(rgb_image, predicted_normals, flow_depth, flow_depth_confidence_scores)

        if self.output_path != '':
            self.visualization(depth_complete, input_batch)

        return depth_complete


    def _get_network_output_depth(self, network_output):
        assert(self._network_estimates_depth())
        return network_output['d']


    def _get_network_output_uncertainty(self, network_output):
        assert(self._network_estimates_depth())
        return network_output['u']


    def _network_loss(self, input_batch, cnn_outputs):
        losses_map = {}
        other_outputs = {}

        _, _, height, width = input_batch['image'].shape
        image_size = height * width

        if self._network_estimates_depth():
            depths_gt = input_batch['depth'].float().cuda(non_blocking=True)
            depth_mask = depths_gt > 0

            depth_loss_func = {
                'L1': torch.nn.L1Loss(reduction='sum'),
                'L2': torch.nn.MSELoss(reduction='sum'),
            }[self.args.depth_loss]
            pred_depth = self._get_network_output_depth(cnn_outputs)

            if self.args.uncertainty_threshold > 0.0:
                pred_uncertainty = self._get_network_output_uncertainty(cnn_outputs)
                for i in range(len(pred_depth)):
                    depth_loss = torch.sum(torch.abs(pred_depth[i][depth_mask] - depths_gt[depth_mask]) * \
                                           torch.exp(-pred_uncertainty[i][depth_mask]) + \
                                           pred_uncertainty[i][depth_mask])
                    losses_map['depth_' + self.args.depth_loss + ('_iter_%d' % i)] \
                        = (1.0 / 1.2 ** (len(pred_depth) - i - 1)) * depth_loss / image_size
            else:
                for i in range(len(pred_depth)):
                    depth_loss = depth_loss_func(pred_depth[i][depth_mask], depths_gt[depth_mask])
                    losses_map['depth_' + self.args.depth_loss + ('_iter_%d' % i)] \
                        = (1.0 / 1.2 ** (len(pred_depth) - i - 1)) * depth_loss / image_size

            # Compute the ratios.
            valid_preds = pred_depth[-1][depth_mask]
            valid_gt = depths_gt[depth_mask]

            printable_ratios = network_run.GetDepthPrintableRatios(valid_gt, valid_preds)
            other_outputs.update(printable_ratios)

        return losses_map, other_outputs


    def _network_evaluate(self, input_batch, cnn_outputs):
        normal_error = None
        depth_ratio_error = None
        depth_abs_error = None

        if self._network_estimates_depth():
            pred_depths = self._get_network_output_depth(cnn_outputs)
            pred_depths = pred_depths[-1]
            depths_gt = input_batch['depth'].cuda()
            depth_mask = (depths_gt > 0).float()
            depth_ratio_np = torch.max(depths_gt / pred_depths, pred_depths / depths_gt).detach().cpu().numpy()

            if self.args.uncertainty_threshold > 0.0:
                pred_uncertainties = self._get_network_output_uncertainty(cnn_outputs)
                depth_mask = (depths_gt > 0.1).float() * \
                             (depths_gt < 10.0).float() * \
                             (torch.exp(pred_uncertainties[-1]) < self.args.uncertainty_threshold).float()

            depth_mask_np = depth_mask.detach().cpu().numpy() > 0
            depth_ratio_error = depth_ratio_np[depth_mask_np]
            depth_abs_error = (depths_gt - pred_depths).abs().detach().cpu().numpy()[depth_mask_np]
        return normal_error, depth_ratio_error, depth_abs_error


if __name__ == '__main__':
    args = ParseCmdLineArguments()
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    network_utils.ConfigureLogging(args.save)

    # First log all the arguments and the values for the record.
    logging.info('sys.argv = {}'.format(sys.argv))
    logging.info('parsed arguments and their values: {}'.format(vars(args)))

    if args.dataset_type == 'scannet':
        train_dataset = ScanNetSmallFramesDepthFlowDataset(usage='train', root=args.root,
                                                  skip_every_n_image=args.skip_every_n_image_train,
                                                  dataset_pickle_file=args.dataset_pickle_file,
                                                  predicted_normal_subdirectory=args.predicted_normal_subdirectory,
                                                  predicted_flowdepth_subdirectory=args.predicted_flowdepth_subdirectory)

        test_dataset = ScanNetSmallFramesDepthFlowDataset(usage='test', root=args.root,
                                                 dataset_pickle_file=args.dataset_pickle_file,
                                                 skip_every_n_image=args.skip_every_n_image_test,
                                                 predicted_normal_subdirectory=args.predicted_normal_subdirectory,
                                                 predicted_flowdepth_subdirectory=args.predicted_flowdepth_subdirectory)
    elif args.dataset_type == 'azure':
        # For now, train and test are the same for azure datasets.
        train_dataset = KinectAzureRescaledDataset(usage='test',
                                           dataset_pickle_file=args.dataset_pickle_file,
                                           skip_every_n_image=args.skip_every_n_image_train,
                                           predicted_flowdepth_subdirectory=args.predicted_flowdepth_subdirectory)
        test_dataset = KinectAzureRescaledDataset(usage='test',
                                          dataset_pickle_file=args.dataset_pickle_file,
                                          skip_every_n_image=args.skip_every_n_image_test,
                                          predicted_flowdepth_subdirectory=args.predicted_flowdepth_subdirectory)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.dataloader_train_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.dataloader_test_workers,
                                 pin_memory=True)


    net_creator = lambda: DRN(args)
    network = RunIterativeDepthCompletion(args, train_dataloader, test_dataloader, network_class_creator=net_creator)


    # Check if this is training or testing.
    if args.train != 0:
        logging.info('Training the network.')
        if args.epoch != 0:
            resume_model = os.path.join(args.save, 'model-epoch-{0:05d}-iter-{1:05d}.ckpt'.format(args.epoch, args.iter))
            network.load_network_from_file(resume_model)
        elif args.checkpoint:
            network.load_network_from_file(args.checkpoint[0])
        if args.surface_normal_checkpoint != '':
            network.load_surface_normal_network_from_file(args.surface_normal_checkpoint)
        else:
            logging.warning('NO CHECKPOINTS LOADED! SURFACE NORMAL IS LOADED FROM PRECOMPUTED FILE.')
        if args.save == '':
            logging.warning('NO CHECKPOINTS WILL BE SAVED! SET --save FLAG TO SAVE TO A DIRECTORY.')
        network.train(starting_epoch=args.epoch, max_epochs=args.max_epochs)
    else:
        assert args.checkpoint is not None
        if args.surface_normal_checkpoint != '':
            network.load_surface_normal_network_from_file(args.surface_normal_checkpoint)
        else:
            logging.warning('NO CHECKPOINTS LOADED! SURFACE NORMAL IS LOADED FROM PRECOMPUTED FILE.')
        for checkpoint in args.checkpoint:
            network.load_network_from_file(checkpoint)
            network.evaluate()
