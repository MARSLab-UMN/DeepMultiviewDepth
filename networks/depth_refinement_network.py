import logging
import torch
import torch.nn as nn
from networks.network_utils import *
from networks.depth_completion import ResNetPyramids, ExtendedFPN


""" 
One iteration update block of the iterative refinement module (IRM), seeks to refine the feature representation
 - Inputs: 
        (1) current feature representation - h; 
        (2) current depth map estimate - [d_est, cvx_est]; 
        (3) triangulated depth and its confidence score - dc_triangulate
 - Outputs:
        (1) updated feature representation - h;   
        (2) updated depth map estimate - [d_update, cvx_update];
"""

class UpdateBlockIRM(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128, depth_input_channel=3):
        super(UpdateBlockIRM, self).__init__()

        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

        # This is independent of triangulated depth input and confidence score, as it aims to learn depth decoder's gradient
        self.convq = nn.Conv2d(hidden_dim+input_dim-depth_input_channel, hidden_dim, 3, padding=1)

        self.depth_decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

        self.cvx_mask_decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 16 * 9, 1, padding=0)
        )

        # Aim to produce weights for the weighted ls problem with all high-level features from everything
        self.weighted_ls_decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1, padding=0),
            nn.Sigmoid()
        )


    def forward(self, h, d_est, cvx_mask_est, dc_triangulate):

        # Unpack the input x
        d_triangulation = dc_triangulate[:, 0:1]
        x = torch.cat([d_est, cvx_mask_est, dc_triangulate], dim=1)

        # Predict the weights for the weighted LS problem
        w_ls = self.weighted_ls_decoder(h)

        # Intermediate results of GRU
        z = torch.sigmoid(self.convz(torch.cat([h, x], dim=1)))
        r = torch.sigmoid(self.convr(torch.cat([h, x], dim=1)))

        # Aim to learn derivative of depth decoder
        q = torch.tanh(self.convq(torch.cat([r*h, d_est, cvx_mask_est], dim=1)))

        # Update
        h = (1-z) * h + z * q * w_ls * (d_triangulation - d_est)

        d_update = self.depth_decoder(h)
        cvx_mask_update = self.cvx_mask_decoder(h)

        return h, d_update, cvx_mask_update


""" 
Depth Refinement Network (DRN) refines depth map from initial triangulated depths and provides uncertainty estimates
- Inputs: 
        (1) rgb image - image; 
        (2) surface normal - normal;
        (3) triangulated depth - depth_flow; 
        (4) and its confidence score - depth_flow_confidence_scores
 - Output: A dictionary containing the following keys:
        (1) 'd': depth maps estimated from all iterations;  
        (2) 'u': uncertainty estimated from all iterations
"""

class DRN(ExtendedFPN):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.args = arguments
        self.feature_concat = None

        self.resnet_depth = ResNetPyramids(in_channels=3, pretrained=False, resnet_arch=self.args.resnet_arch)

        self.depth_feature_concat = nn.Sequential(
            nn.Conv2d(self.input_channels[0] // 2 * self.num_encoders, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

        self.depth_upsampling = nn.Sequential(
            nn.Conv2d(self.input_channels[0] // 2 * self.num_encoders, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 16 * 9, 1)
        )

        self.depth_output_uncertainty = nn.Sequential(
            nn.Conv2d(self.input_channels[0] // 2 * self.num_encoders, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Conv2d(64, 1, 1)
        )

        self.update_block = UpdateBlockIRM(hidden_dim=self.input_channels[0] // 2 * self.num_encoders,
                                           input_dim=16*9+4)

        if self.__class__.__name__ == 'DRN':
            logging.info("Backbone: ResNet-{}. Number of parameters in model: {}".format(self.args.resnet_arch,
                                                                                        count_parameters(self)))


    # Upsample depth [H/4, W/4, 1] -> [H, W, 1] using convex combination
    @staticmethod
    def upsample_depth(depth, mask):
        N, _, H, W = depth.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_depth = F.unfold(depth, [3, 3], padding=1)
        up_depth = up_depth.view(N, 1, 9, 1, 1, H, W)

        up_depth = torch.sum(mask * up_depth, dim=2)
        up_depth = up_depth.permute(0, 1, 4, 2, 5, 3)
        return up_depth.reshape(N, 1, 4 * H, 4 * W)


    @staticmethod
    def combine_rgbd_features(rgb, normal, depth):
        return torch.cat((rgb, normal, depth), dim=1)


    def forward(self, image, normal, depth_flow, depth_flow_confidence_scores):
        # Feature extraction
        i1, i2, i3, i4 = self.resnet_rgb(image)
        n1, n2, n3, n4 = self.resnet_normal(normal)

        depth_input = torch.cat((depth_flow, depth_flow_confidence_scores), dim=1)
        d1, d2, d3, d4 = self.resnet_depth(depth_input)

        z1 = self.feature1_upsampling(self.combine_rgbd_features(i1, n1, d1))
        z2 = self.feature2_upsampling(self.combine_rgbd_features(i2, n2, d2))
        z3 = self.feature3_upsampling(self.combine_rgbd_features(i3, n3, d3))
        z4 = self.feature4_upsampling(self.combine_rgbd_features(i4, n4, d4))
        h_0 = z1 + z2 + z3 + z4

        # Initial depth estimate
        d_0 = self.depth_feature_concat(h_0)
        m_0 = self.depth_upsampling(h_0)
        u_pred_0 = self.depth_output_uncertainty(h_0)
        d_pred_0 = self.upsample_depth(depth=d_0, mask=m_0)

        d_pred = []
        u_pred = []
        d_pred.append(d_pred_0)
        u_pred.append(u_pred_0)

        # Prepare iterative inputs
        depth_flow_small = torch.nn.functional.interpolate(depth_flow, scale_factor=1 / 4).detach()
        depth_flow_confidence_scores_small = torch.nn.functional.interpolate(depth_flow_confidence_scores, scale_factor=1 / 4,
                                                                       mode='bilinear', align_corners=False).detach()
        depth_input_small = depth_flow_small
        depth_input_small = torch.cat((depth_input_small, depth_flow_confidence_scores_small), dim=1)

        d_i = d_0
        h_i = h_0
        m_i = m_0

        # Iterative update
        for i in range(self.args.refinement_iterations):
            h_update, d_ip, m_ip = self.update_block(h_i, d_i, m_i, depth_input_small)

            d_pred_i = self.upsample_depth(depth=d_ip, mask=m_ip)
            d_pred.append(d_pred_i)

            # updating
            h_i = h_update
            d_i = d_ip
            m_i = m_ip
            u_pred.append(self.depth_output_uncertainty(h_i))

        return {'d': d_pred, 'u': u_pred}
