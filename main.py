import argparse
import logging
import numpy as np
import os
import sys
import torch
import torch.autograd
from torch.utils.data import DataLoader
from PIL import Image

from demo_dataset import DemoFlowDataset, NyuFlowDataset
import network_run
from networks.depth_refinement_network import DRN

from networks.surface_normal import *
from networks.surface_normal_dorn import *
import networks.network_utils as network_utils

from run_raft import RunDepthRAFT
from run_drn import RunIterativeDepthCompletion

sys.path.append('networks/raft/core')
from networks.raft.core.raft import RAFT


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
    parser.add_argument('--metrics_averaged_among_images', type=int, default=1,
                        help='Which type of metric we are computing.')

    # For multiviews
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

    # Depth Refinement Network
    parser.add_argument('--drn_model', type=str, default='')
    parser.add_argument('--refinement_iterations', type=int, default=5,
                        help='Iterative refinement iterations')
    parser.add_argument('--uncertainty_threshold', type=float, default=-1.0,
                        help='Depth uncertainty thresholds.')
    return parser.parse_args()


class RunMultiViewDepthEstimation(RunDepthRAFT, RunIterativeDepthCompletion):
    def __init__(self, arguments, train_dataloader, test_dataloader, network_class_creator):
        super().__init__(arguments, train_dataloader, test_dataloader, network_class_creator=network_class_creator)
        if self.args.surface_normal_checkpoint != '':
            self.surface_normal_cnn = self.surface_normal_cnn = SurfaceNormalDORN().cuda()
        else:
            self.surface_normal_cnn = None
        self.depth_refinement_network = DRN(arguments).cuda()

        # Visualization
        self.output_path = arguments.save_visualization
        if self.output_path != '':
            if not os.path.exists(self.output_path):
                os.mkdir(self.output_path)
        self.save_idx = 0

    def load_depth_refinement_network_from_file(self, checkpoint):
        state = self.depth_refinement_network.state_dict()
        state.update(torch.load(checkpoint))
        self.depth_refinement_network.load_state_dict(state)
        self.depth_refinement_network.eval()

    def load_surface_normal_network_from_file(self, checkpoint):
        state = self.surface_normal_cnn.state_dict()
        state.update(torch.load(checkpoint))
        self.surface_normal_cnn.load_state_dict(state)
        self.surface_normal_cnn.eval()

    def visualization(self, output_pred, flow_depth, input_batch):
        output_path = self.output_path

        # rgb images
        rgb = input_batch['image'][0].squeeze().detach().cpu().numpy()
        rgb_image = np.transpose(rgb, axes=[1, 2, 0]).astype(np.uint8)

        depth_flow_img = self.create_color_depth_image(
            flow_depth[0].squeeze().detach().cpu().numpy(),
            thres=5.0)

        depth_gt_img = self.create_color_depth_image(input_batch['depth'][0].squeeze().detach().cpu().numpy(),
                                                     thres=5.0)
        depth_pred_img = self.create_color_depth_image(output_pred['d'][-1][0].squeeze().detach().cpu().numpy(),
                                                       thres=5.0)
        depth_error_img = self.create_color_error_depth_image(
            (output_pred['d'][-1][0].cpu() - input_batch['depth'][0]).abs().squeeze().detach().cpu().numpy(),
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
        image.save(os.path.join(output_path, 'viz_{0:06d}.png'.format(input_batch['frame_index'][0])))

    def _call_cnn(self, input_batch):
        bearings_ref_in_other, image1, image2, ts = self.load_inputs(input_batch)
        rgb_image = input_batch['imagen'].cuda(non_blocking=True)

        # import time
        # start_time = time.time()

        if image1.shape[0] == 1:
            # When inference for a single image, put all pairs in a batch.
            flows_batch = self.get_flow(image1.expand(image2.shape[1], -1, -1, -1), image2[0])
            flows = [flow[None, ...] for flow in flows_batch]
        else:
            flows = [self.get_flow(image1, image2[:, k, ...]) for k in range(image2.shape[1])]

        flow_depth, flow_depth_confidence_scores = self.triangulation(bearings_ref_in_other, ts, flows, residual=True)
        flow_depth_confidence_scores = torch.cat(flow_depth_confidence_scores, dim=1)

        # flow_time = time.time() - start_time
        # start_time = time.time()

        with torch.no_grad():
            if self.surface_normal_cnn is not None:
                predicted_normals = self.surface_normal_cnn(rgb_image)
            else:
                predicted_normals = input_batch['predicted_normal'].cuda(non_blocking=True)
            predicted_normals = torch.nn.functional.normalize(predicted_normals)

        # normal_time = time.time() - start_time
        # start_time = time.time()

        # rescale to ScanNet
        flow_depth_s, flow_depth_confidence_scores_s = [
            torch.nn.functional.interpolate(f[:, :, 13:-12, 17:-16], size=(240, 320),
                                            mode='nearest') for f in [flow_depth, flow_depth_confidence_scores]]
        depth_complete = self.depth_refinement_network(rgb_image, predicted_normals,
                                                       flow_depth_s, flow_depth_confidence_scores_s)
        # depth_complete = self.depth_refinement_network(rgb_image, predicted_normals,
        #                                                flow_depth, flow_depth_confidence_scores)

        # rescale back
        final = {key: 3 * torch.ones_like(depth_complete[key][-1]) for key in depth_complete}
        final['d'] = flow_depth
        for key in depth_complete:
            final[key][:, :, 13:-12, 17:-16] = torch.nn.functional.interpolate(depth_complete[key][-1], size=(215, 287),
                                                                               mode='nearest')
            final[key] = [final[key]]
        return final

        # drn_time = time.time() - start_time
        # print(flow_time, normal_time, drn_time, sep='\t')

        # if self.output_path != '':
        #     self.visualize(depth_complete, flow_depth, input_batch, normal_pred=predicted_normals)
        # return depth_complete

    def _network_evaluate(self, input_batch, cnn_outputs):
        normal_error = None
        depth_ratio_error = None
        depth_abs_error = None
        depth_error_every_image = None

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

            if self.args.metrics_averaged_among_images:
                all_abs_rel, all_sq_rel, all_log_rmse, all_i_rmse, all_si_log = [], [], [], [], []
                all_mad, all_rmse, all_a05, all_a10, all_a1, all_a2, all_a3 = [], [], [], [], [], [], []

                depth_abs_error_no_mask = (depths_gt - pred_depths).abs().detach().cpu().numpy()
                depth_ratio_error_no_mask = depth_ratio_np
                for i in range(depth_abs_error_no_mask.shape[0]):
                    if depth_mask_np[i].sum() > 0:
                        depth_abs_error_image_i = depth_abs_error_no_mask[i][depth_mask_np[i]]
                        depth_ratio_error_image_i = depth_ratio_error_no_mask[i][depth_mask_np[i]]

                        # additional metrics
                        pr = pred_depths[i, ...].squeeze().detach().cpu().numpy()[depth_mask_np[i].squeeze()]
                        gt = depths_gt[i, ...].squeeze().detach().cpu().numpy()[depth_mask_np[i].squeeze()]

                        abs_rel = np.mean(np.abs(gt - pr) / gt)
                        sq_rel = np.mean(((gt - pr) ** 2) / gt)
                        rmse_log = (np.log(gt) - np.log(pr)) ** 2
                        rmse_log = np.sqrt(rmse_log.mean())
                        i_rmse = (1 / gt - 1 / (pr + 1e-4)) ** 2
                        i_rmse = np.sqrt(i_rmse.mean())

                        # sc_inv
                        log_diff = np.log(gt) - np.log(pr)
                        num_pixels = np.float32(log_diff.size)
                        sc_inv = np.sqrt(
                            np.sum(np.square(log_diff)) / num_pixels - np.square(np.sum(log_diff)) / np.square(
                                num_pixels))

                        all_abs_rel.append(abs_rel)
                        all_sq_rel.append(sq_rel)
                        all_log_rmse.append(rmse_log)
                        all_i_rmse.append(i_rmse)
                        all_si_log.append(sc_inv)

                        all_mad.append(np.mean(depth_abs_error_image_i).item())
                        all_rmse.append(np.sqrt(np.mean(depth_abs_error_image_i ** 2)).item())

                        all_a05.append(
                            100. * np.sum(depth_ratio_error_image_i < 1.05).item() / depth_ratio_error_image_i.shape[0])
                        all_a10.append(
                            100. * np.sum(depth_ratio_error_image_i < 1.10).item() / depth_ratio_error_image_i.shape[0])
                        all_a1.append(
                            100. * np.sum(depth_ratio_error_image_i < 1.25).item() / depth_ratio_error_image_i.shape[0])
                        all_a2.append(100. * np.sum(depth_ratio_error_image_i < 1.25 ** 2).item() /
                                      depth_ratio_error_image_i.shape[0])
                        all_a3.append(100. * np.sum(depth_ratio_error_image_i < 1.25 ** 3).item() /
                                      depth_ratio_error_image_i.shape[0])

                depth_error_every_image = {'abs_rel': np.array(all_abs_rel),
                                           'sq_rel': np.array(all_sq_rel),
                                           'rmse_log': np.array(all_log_rmse),
                                           'i_rmse': np.array(all_i_rmse),
                                           'sc_inv': np.array(all_si_log),
                                           'mad': np.array(all_mad),
                                           'rmse': np.array(all_rmse),
                                           '1.05': np.array(all_a05),
                                           '1.1': np.array(all_a10),
                                           '1.25': np.array(all_a1),
                                           '1.25^2': np.array(all_a2),
                                           '1.25^3': np.array(all_a3)}
                return depth_error_every_image
            else:
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

    if args.dataset_type == 'nyu':
        train_dataset = NyuFlowDataset(args.root, 'train', args.dataset_pickle_file, args.window,
                                       skip_every_n_image=args.skip_every_n_image_train)
        test_dataset = NyuFlowDataset(args.root, 'test', args.dataset_pickle_file, args.window,
                                      skip_every_n_image=args.skip_every_n_image_test)
    else:
        train_dataset = DemoFlowDataset(usage='train', window=args.window, root='./dataset')

        test_dataset = DemoFlowDataset(usage='test', window=args.window, root='./dataset')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.dataloader_train_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.dataloader_test_workers,
                                 pin_memory=True)

    network = RunMultiViewDepthEstimation(args, train_dataloader, test_dataloader,
                                          network_class_creator=(lambda: RAFT(args)))

    # Check if this is training or testing.
    assert args.raft_model is not ''
    assert args.drn_model is not ''
    if args.surface_normal_checkpoint != '':
        network.load_surface_normal_network_from_file(args.surface_normal_checkpoint)
    else:
        logging.warning('NO CHECKPOINTS LOADED! SURFACE NORMAL IS LOADED FROM PRECOMPUTED FILE.')

    network.load_network_from_file(args.raft_model)
    network.load_depth_refinement_network_from_file(args.drn_model)
    network.evaluate(metrics_averaged_among_images=args.metrics_averaged_among_images)
