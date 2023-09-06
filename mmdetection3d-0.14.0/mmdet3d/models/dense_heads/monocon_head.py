import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils.gaussian_target import (gaussian_radius, gen_gaussian_target)
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                                transpose_and_gather_feat)
from mmdet3d.ops.attentive_norm import AttnBatchNorm2d

INF = 1e8
EPS = 1e-12
PI = np.pi


@HEADS.register_module()
class MonoConHead(nn.Module):
    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 bbox3d_code_size=7,
                 num_kpt=9,
                 num_alpha_bins=12,
                 max_objs=30,
                 vector_regression_level=1,
                 pred_bbox2d=True,
                 loss_center_heatmap=None,
                 loss_wh=None,
                 loss_offset=None,
                 loss_center2kpt_offset=None,
                 loss_kpt_heatmap=None,
                 loss_kpt_heatmap_offset=None,
                 loss_dim=None,
                 loss_depth=None,
                 loss_alpha_cls=None,
                 loss_alpha_reg=None,
                 multibins=False,
                 use_AN=True,
                 num_AN_affine=10,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(MonoConHead, self).__init__()
        assert bbox3d_code_size >= 7
        self.num_classes = num_classes
        self.bbox_code_size = bbox3d_code_size
        self.pred_bbox2d = pred_bbox2d
        self.max_objs = max_objs
        self.num_kpt = num_kpt
        self.num_alpha_bins = num_alpha_bins
        self.vector_regression_level = vector_regression_level

        self.use_AN = use_AN
        self.num_AN_affine = num_AN_affine
        self.norm = AttnBatchNorm2d if use_AN else nn.BatchNorm2d

        self.heatmap_head = self._build_head(in_channel, feat_channel, num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)
        self.center2kpt_offset_head = self._build_head(in_channel, feat_channel, self.num_kpt * 2)
        self.kpt_heatmap_head = self._build_head(in_channel, feat_channel, self.num_kpt)
        self.kpt_heatmap_offset_head = self._build_head(in_channel, feat_channel, 2)
        self.dim_head = self._build_head(in_channel, feat_channel, 3)
        self.depth_head = self._build_head(in_channel, feat_channel, 2)
        self.multibins = multibins
        self._build_dir_head(in_channel, feat_channel)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        self.loss_center2kpt_offset = build_loss(loss_center2kpt_offset)
        self.loss_kpt_heatmap = build_loss(loss_kpt_heatmap)
        self.loss_kpt_heatmap_offset = build_loss(loss_kpt_heatmap_offset)
        self.loss_dim = build_loss(loss_dim)
        if 'Aware' in loss_dim['type']:
            self.dim_aware_in_loss = True
        else:
            self.dim_aware_in_loss = False
        self.loss_depth = build_loss(loss_depth)
        self.loss_alpha_cls = build_loss(loss_alpha_cls)
        self.loss_alpha_reg = build_loss(loss_alpha_reg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        # multibins
        self.angle_kl_target = self.cal_conf()
        self.criterion = nn.KLDivLoss(reduction="mean")

    def cal_conf(self):
        kl_target = [None] * self.num_alpha_bins
        for i in range(self.num_alpha_bins):
            conf = [None] * self.num_alpha_bins
            for j in range(self.num_alpha_bins):
                conf[j] = (
                    1
                    / (np.sqrt(2 * PI) * 1)
                    * np.exp(
                        -(
                            (
                                (j + self.num_alpha_bins / 2 - i)
                                % (self.num_alpha_bins)
                                - self.num_alpha_bins / 2
                            )
                            ** 2
                        )
                        / 2
                        / (1**2)
                    )
                )
            kl_target[i] = conf
        return kl_target    

    def _build_head(self, in_channel, feat_channel, out_channel):
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            self._get_norm_layer(feat_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def _build_dir_head(self, in_channel, feat_channel):
        self.dir_feat = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            self._get_norm_layer(feat_channel),
            nn.ReLU(inplace=True),
        )
        self.dir_cls = nn.Sequential(nn.Conv2d(feat_channel, self.num_alpha_bins, kernel_size=1))
        self.dir_reg = nn.Sequential(nn.Conv2d(feat_channel, self.num_alpha_bins, kernel_size=1))
        if self.multibins:
            self.dir_kl = nn.Sequential(nn.Conv2d(feat_channel, self.num_alpha_bins, kernel_size=1))


    def _get_norm_layer(self, feat_channel):
        return self.norm(feat_channel, momentum=0.03, eps=0.001) if not self.use_AN else \
            self.norm(feat_channel, self.num_AN_affine, momentum=0.03, eps=0.001)

    def init_weights(self):
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)  # -2.19
        self.kpt_heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head, self.center2kpt_offset_head, self.depth_head,
                     self.kpt_heatmap_offset_head, self.dim_head, self.dir_feat,
                     self.dir_cls, self.dir_reg]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
        if self.multibins:
            for head in [self.dir_kl]:
                for m in head.modules():
                    if isinstance(m, nn.Conv2d):
                        normal_init(m, std=0.001)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        #import ipdb; ipdb.set_trace()
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        center_heatmap_pred = torch.clamp(center_heatmap_pred, min=1e-4, max=1 - 1e-4)
        kpt_heatmap_pred = self.kpt_heatmap_head(feat).sigmoid()
        kpt_heatmap_pred = torch.clamp(kpt_heatmap_pred, min=1e-4, max=1 - 1e-4)

        offset_pred = self.offset_head(feat)
        kpt_heatmap_offset_pred = self.kpt_heatmap_offset_head(feat)

        wh_pred = self.wh_head(feat)
        center2kpt_offset_pred = self.center2kpt_offset_head(feat)
        dim_pred = self.dim_head(feat)
        depth_pred = self.depth_head(feat)
        depth_pred[:, 0, :, :] = 1. / (depth_pred[:, 0, :, :].sigmoid() + EPS) - 1

        alpha_feat = self.dir_feat(feat)
        alpha_cls_pred = self.dir_cls(alpha_feat)
        alpha_offset_pred = self.dir_reg(alpha_feat)
        alpha_kl_pred = None
        # multibins
        if self.multibins:
            alpha_kl_pred = self.dir_kl(alpha_feat)
        return center_heatmap_pred, wh_pred, offset_pred, center2kpt_offset_pred, kpt_heatmap_pred, \
               kpt_heatmap_offset_pred, dim_pred, alpha_cls_pred, alpha_offset_pred, depth_pred, alpha_kl_pred

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds', 'center2kpt_offset_preds',
                          'kpt_heatmap_preds', 'kpt_heatmap_offset_preds', 'dim_preds', 'alpha_cls_preds',
                          'alpha_offset_preds', 'depth_preds'))
    def loss(self,
             center_heatmap_preds,
             wh_preds,
             offset_preds,
             center2kpt_offset_preds,
             kpt_heatmap_preds,
             kpt_heatmap_offset_preds,
             dim_preds,
             alpha_cls_preds,
             alpha_offset_preds,
             depth_preds,
             alpha_kl_preds,
             gt_bboxes,
             gt_labels,
             gt_bboxes_3d,
             gt_labels_3d,
             centers2d,
             depths,
             gt_kpts_2d,
             gt_kpts_valid_mask,
             img_metas,
             attr_labels=None,
             proposal_cfg=None,
             gt_bboxes_ignore=None):

        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) \
               == len(center2kpt_offset_preds) == len(kpt_heatmap_preds) == len(kpt_heatmap_offset_preds) \
               == len(dim_preds) == len(alpha_cls_preds) == len(alpha_offset_preds) == 1
        #import ipdb; ipdb.set_trace()
        center_heatmap_pred = center_heatmap_preds[0]
        wh_pred = wh_preds[0]
        offset_pred = offset_preds[0]
        center2kpt_offset_pred = center2kpt_offset_preds[0]
        kpt_heatmap_pred = kpt_heatmap_preds[0]
        kpt_heatmap_offset_pred = kpt_heatmap_offset_preds[0]
        dim_pred = dim_preds[0]
        alpha_cls_pred = alpha_cls_preds[0]
        alpha_offset_pred = alpha_offset_preds[0]
        depth_pred = depth_preds[0]
        
        batch_size = center_heatmap_pred.shape[0]

        target_result = self.get_targets(gt_bboxes, gt_labels,
                                         gt_bboxes_3d,
                                         centers2d,
                                         depths,
                                         gt_kpts_2d,
                                         gt_kpts_valid_mask,
                                         center_heatmap_pred.shape,
                                         img_metas[0]['pad_shape'],
                                         img_metas)

        # 2D
        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        # 2D-3D 3D
        center2kpt_offset_target = target_result['center2kpt_offset_target']
        dim_target = target_result['dim_target']
        depth_target = target_result['depth_target']
        alpha_cls_target = target_result['alpha_cls_target']
        alpha_offset_target = target_result['alpha_offset_target']
        kpt_heatmap_target = target_result['kpt_heatmap_target']
        kpt_heatmap_offset_target = target_result['kpt_heatmap_offset_target']

        indices = target_result['indices']
        indices_kpt = target_result['indices_kpt']

        mask_target = target_result['mask_target']
        mask_center2kpt_offset = target_result['mask_center2kpt_offset']
        mask_kpt_heatmap_offset = target_result['mask_kpt_heatmap_offset']
        # multibins
        class_id_target = target_result['class_id_target']
        loc_target = target_result['loc_target']
        yaw_target = target_result["yaw_target"]
        # select desired preds and labels based on mask

        # 2d offset
        offset_pred = self.extract_input_from_tensor(offset_pred, indices, mask_target)
        offset_target = self.extract_target_from_tensor(offset_target, mask_target)
        # 2d size
        wh_pred = self.extract_input_from_tensor(wh_pred, indices, mask_target)
        wh_target = self.extract_target_from_tensor(wh_target, mask_target)
        # 3d dim
        dim_pred = self.extract_input_from_tensor(dim_pred, indices, mask_target)
        dim_target = self.extract_target_from_tensor(dim_target, mask_target)
        # depth
        depth_pred = self.extract_input_from_tensor(depth_pred, indices, mask_target)
        depth_target = self.extract_target_from_tensor(depth_target, mask_target)
        #import ipdb; ipdb.set_trace()
        # alpha cls
        alpha_cls_pred = self.extract_input_from_tensor(alpha_cls_pred, indices, mask_target)
        alpha_cls_target = self.extract_target_from_tensor(alpha_cls_target, mask_target).type(torch.long)
        alpha_cls_onehot_target = alpha_cls_target.new_zeros([len(alpha_cls_target), self.num_alpha_bins]).scatter_(
            dim=1, index=alpha_cls_target.view(-1, 1), value=1)
                
        #import ipdb; ipdb.set_trace()
        # alpha offset
        alpha_offset_pred = self.extract_input_from_tensor(alpha_offset_pred, indices, mask_target)
        alpha_offset_pred = torch.sum(alpha_offset_pred * alpha_cls_onehot_target, 1, keepdim=True)
        alpha_offset_target = self.extract_target_from_tensor(alpha_offset_target, mask_target)

        if mask_target.sum() > 0:
            loss_alpha_cls = self.loss_alpha_cls(alpha_cls_pred, alpha_cls_onehot_target)
        else:
            loss_alpha_cls = 0.0
        loss_alpha_reg = self.loss_alpha_reg(alpha_offset_pred, alpha_offset_target)
        
        # center2kpt offset
        center2kpt_offset_pred = self.extract_input_from_tensor(center2kpt_offset_pred,
                                                                indices, mask_target)  # B * (num_kpt * 2)
        center2kpt_offset_target = self.extract_target_from_tensor(center2kpt_offset_target, mask_target)
        mask_center2kpt_offset = self.extract_target_from_tensor(mask_center2kpt_offset, mask_target)
        center2kpt_offset_pred *= mask_center2kpt_offset
        loss_center2kpt_offset = self.loss_center2kpt_offset(center2kpt_offset_pred, center2kpt_offset_target,
                                                             avg_factor=(mask_center2kpt_offset.sum() + EPS))

        # kpt heatmap offset
        kpt_heatmap_offset_pred = transpose_and_gather_feat(kpt_heatmap_offset_pred, indices_kpt)
        kpt_heatmap_offset_pred = kpt_heatmap_offset_pred.reshape(batch_size, self.max_objs, self.num_kpt * 2)
        kpt_heatmap_offset_pred = kpt_heatmap_offset_pred[mask_target]
        kpt_heatmap_offset_target = kpt_heatmap_offset_target[mask_target]
        mask_kpt_heatmap_offset = self.extract_target_from_tensor(mask_kpt_heatmap_offset, mask_target)

        kpt_heatmap_offset_pred *= mask_kpt_heatmap_offset
        loss_kpt_heatmap_offset = self.loss_kpt_heatmap_offset(kpt_heatmap_offset_pred, kpt_heatmap_offset_target,
                                                               avg_factor=(mask_kpt_heatmap_offset.sum() + EPS))

        # calculate loss
        loss_center_heatmap = self.loss_center_heatmap(center_heatmap_pred, center_heatmap_target)
        loss_kpt_heatmap = self.loss_kpt_heatmap(kpt_heatmap_pred, kpt_heatmap_target)

        loss_wh = self.loss_wh(wh_pred, wh_target)
        loss_offset = self.loss_offset(offset_pred, offset_target)
        if self.dim_aware_in_loss:
            loss_dim = self.loss_dim(dim_pred, dim_target, dim_pred)
        else:
            loss_dim = self.loss_dim(dim_pred, dim_target)

        depth_pred, depth_log_variance = depth_pred[:, 0:1], depth_pred[:, 1:2]
        loss_depth = self.loss_depth(depth_pred, depth_target, depth_log_variance)
    
        losses = dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset,
            loss_dim=loss_dim,
            loss_center2kpt_offset=loss_center2kpt_offset,
            loss_kpt_heatmap=loss_kpt_heatmap,
            loss_kpt_heatmap_offset=loss_kpt_heatmap_offset,
            loss_alpha_cls=loss_alpha_cls,
            loss_alpha_reg=loss_alpha_reg,
            loss_depth=loss_depth,
        )

        if self.multibins:
            #import ipdb; ipdb.set_trace()
            #class_id_target = self.extract_target_from_tensor(class_id_target, mask_target).type(torch.long)
            class_id_target = alpha_cls_target
            loc_target = self.extract_target_from_tensor(loc_target, mask_target).type(torch.long)
            alpha_kl_target = [None] * class_id_target.shape[0]
            for i in range(class_id_target.shape[0]):
                alpha_kl_target[i] = self.angle_kl_target[
                    class_id_target[i][0].to(torch.int64).cpu()
                ]
            alpha_kl_target = (
                torch.tensor(np.array(alpha_kl_target))
                .to(torch.float)
                .to(target_result["class_id_target"].device)
            )
            alpha_kl_pred = alpha_kl_preds[0]
            alpha_kl_pred = self.extract_input_from_tensor(alpha_kl_pred, indices, mask_target)
            conf = (
                1
                / (np.sqrt(2 * PI) * 1)
                * torch.exp(-((alpha_kl_pred) ** 2) / 2 / (1**2))
            )
            conf = F.softmax(conf, dim=1)
            conf = torch.log(conf + EPS)
            #import ipdb; ipdb.set_trace()
            loss_alpha_kl_distribute = self.criterion(conf, alpha_kl_target)
            
            #import ipdb; ipdb.set_trace()
            #yaw_target = self.extract_target_from_tensor(yaw_target, mask_target)
            #yaw_pred = self.class2angle_kl_pred(alpha_kl_pred, loc_target)
            #import ipdb; ipdb.set_trace()
            #yaw_target[yaw_target > PI] = yaw_target[yaw_target > PI] - 2 * PI
            #yaw_target[yaw_target < -PI] = yaw_target[yaw_target < -PI] + 2 * PI
            #loss_alpha_l1 = self.loss_alpha_reg(yaw_pred, yaw_target)
        
            losses["loss_alpha_kl_distribute"] = loss_alpha_kl_distribute
            #losses["loss_alpha_l1"] = loss_alpha_l1

        return losses 

    def class2angle_kl_pred(self, alpha_kl_pred, loc_target):
        angle_res = None
        # transform net output into gaussian distribute
        conf = (
            1
            / (np.sqrt(2 * PI) * 1)
            * torch.exp(-((alpha_kl_pred) ** 2) / 2 / (1**2))
        )
        #conf = conf / torch.linalg.norm(conf, ord=1, keepdims=True, axis=1)
        conf = F.softmax(conf, dim=1) + 1e-8
        # process the boundary angle
        angle_per_class = 2 * PI / float(self.num_alpha_bins)
        left_index = PI / (3 * angle_per_class)
        right_index = self.num_alpha_bins - left_index
        shift_angle_mask = (conf.argmax(dim=-1) > right_index) | (
            conf.argmax(dim=-1) < left_index
        )
        ori_angle_mask = (conf.argmax(dim=-1) >= left_index) & (
            conf.argmax(dim=-1) <= right_index
        )

        angle_values = (
            torch.arange(0, conf.shape[-1], dtype=torch.float32, device=conf.device)
            * angle_per_class
        )
    
        angle_values = angle_values.unsqueeze(0).expand_as(conf)

        shift_angle_values = (angle_values + PI) % (2 * PI)
        ori_angle_values = angle_values

        conf_shift = (
            conf[shift_angle_mask].clone() * shift_angle_values[shift_angle_mask]
        )
        conf_ori = conf[ori_angle_mask].clone() * ori_angle_values[ori_angle_mask]

        conf[shift_angle_mask] = conf_shift
        conf[ori_angle_mask] = conf_ori

        # get the alpha
        angle_res = conf.sum(dim=-1)
        angle_res[shift_angle_mask] = angle_res[shift_angle_mask] - PI
        angle_res = angle_res % (2 * PI)

        # alpha: -pi ~ pi
        angle_res[angle_res > PI] = angle_res[angle_res > PI] - 2 * PI
        angle_res[angle_res < -PI] = angle_res[angle_res < -PI] + 2 * PI
        rays = torch.atan(loc_target[:, 0] / (loc_target[:, 2] + 1e-7))
        angle_res += rays
        angle_res[angle_res > PI] = angle_res[angle_res > PI] - 2 * PI
        angle_res[angle_res < -PI] = angle_res[angle_res < -PI] + 2 * PI
        angle_res = angle_res.unsqueeze(-1)
        return angle_res

    def class2angle_kl(self, kpts, alpha_kl_pred, calib):
        angle_res = None
        # transform net output into gaussian distribute
        conf = (
            1
            / (np.sqrt(2 * PI) * 1)
            * torch.exp(-((alpha_kl_pred) ** 2) / 2 / (1**2))
        )
        #conf = conf / torch.linalg.norm(conf, ord=1, keepdims=True, axis=1)
        conf = F.softmax(conf, dim=1) + 1e-8
        # process the boundary angle
        angle_per_class = 2 * PI / float(self.num_alpha_bins)
        left_index = PI / (3 * angle_per_class)
        right_index = self.num_alpha_bins - left_index
        shift_angle_mask = (conf.argmax(dim=-1) > right_index) | (
            conf.argmax(dim=-1) < left_index
        )
        ori_angle_mask = (conf.argmax(dim=-1) >= left_index) & (
            conf.argmax(dim=-1) <= right_index
        )

        angle_values = (
            torch.arange(0, conf.shape[-1], dtype=torch.float32, device=conf.device)
            * angle_per_class
        )
    
        angle_values = angle_values.unsqueeze(0).expand_as(conf)

        shift_angle_values = (angle_values + PI) % (2 * PI)
        ori_angle_values = angle_values

        conf_shift = (
            conf[shift_angle_mask].clone() * shift_angle_values[shift_angle_mask]
        )
        conf_ori = conf[ori_angle_mask].clone() * ori_angle_values[ori_angle_mask]

        conf[shift_angle_mask] = conf_shift
        conf[ori_angle_mask] = conf_ori

        # get the alpha
        angle_res = conf.sum(dim=-1)
        angle_res[shift_angle_mask] = angle_res[shift_angle_mask] - PI
        angle_res = angle_res % (2 * PI)

        # alpha: -pi ~ pi
        angle_res[angle_res > PI] = angle_res[angle_res > PI] - 2 * PI
        angle_res[angle_res < -PI] = angle_res[angle_res < -PI] + 2 * PI
        angle_res = angle_res.unsqueeze(-1)
        # get the yaw
        device = kpts.device
        calib = torch.tensor(calib).type(torch.FloatTensor).to(device).unsqueeze(0)
        #import ipdb; ipdb.set_trace()
        si = torch.zeros_like(kpts[:, :, 0:1]) + calib[:, 0:1, 0:1]
        rot_y = angle_res + torch.atan2(kpts[:, :, 0:1] - calib[:, 0:1, 2:3], si)

        while (rot_y > PI).any():
            rot_y[rot_y > PI] = rot_y[rot_y > PI] - 2 * PI
        while (rot_y < -PI).any():
            rot_y[rot_y < -PI] = rot_y[rot_y < -PI] + 2 * PI

        return rot_y


    def get_targets(self, gt_bboxes, gt_labels,
                    gt_bboxes_3d,
                    centers2d,
                    depths,
                    gt_kpts_2d,
                    gt_kpts_valid_mask,
                    feat_shape, img_shape,
                    img_metas):
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        calibs = []

        # objects as 2D center points
        center_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_classes, feat_h, feat_w])

        # 2D attributes
        wh_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 2])
        offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 2])

        # 3D attributes
        dim_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 3])
        alpha_cls_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])
        alpha_offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])
        depth_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])
        # multibins
        class_id_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])
        yaw_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])
        loc_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 3])

        # 2D-3D kpt heatmap and offset
        center2kpt_offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt * 2])
        kpt_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_kpt, feat_h, feat_w])
        kpt_heatmap_offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt * 2])

        # indices
        indices = gt_bboxes[-1].new_zeros([bs, self.max_objs]).type(torch.cuda.LongTensor)
        indices_kpt = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt]).type(torch.cuda.LongTensor)

        # masks
        mask_target = gt_bboxes[-1].new_zeros([bs, self.max_objs])
        mask_center2kpt_offset = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt * 2])
        mask_kpt_heatmap_offset = gt_bboxes[-1].new_zeros([bs, self.max_objs, self.num_kpt * 2])
        #import ipdb; ipdb.set_trace()
        for batch_id in range(bs):
            img_meta = img_metas[batch_id]
            cam_p2 = img_meta['cam_intrinsic']

            gt_bbox = gt_bboxes[batch_id]
            calibs.append(cam_p2)
            if len(gt_bbox) < 1:
                continue
            gt_label = gt_labels[batch_id]
            gt_bbox_3d = gt_bboxes_3d[batch_id]
            if not isinstance(gt_bbox_3d, torch.Tensor):
                gt_bbox_3d = gt_bbox_3d.tensor.to(gt_bbox.device)

            depth = depths[batch_id]

            gt_kpt_2d = gt_kpts_2d[batch_id]
            gt_kpt_valid_mask = gt_kpts_valid_mask[batch_id]

            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            gt_kpt_2d = gt_kpt_2d.reshape(-1, self.num_kpt, 2)
            gt_kpt_2d[:, :, 0] *= width_ratio
            gt_kpt_2d[:, :, 1] *= height_ratio

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio

                dim = gt_bbox_3d[j][3: 6]
                alpha = gt_bbox_3d[j][6]
                gt_kpt_2d_single = gt_kpt_2d[j]  # (9, 2)
                gt_kpt_valid_mask_single = gt_kpt_valid_mask[j]  # (9,)

                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                indices[batch_id, j] = cty_int * feat_w + ctx_int

                wh_target[batch_id, j, 0] = scale_box_w
                wh_target[batch_id, j, 1] = scale_box_h
                offset_target[batch_id, j, 0] = ctx - ctx_int
                offset_target[batch_id, j, 1] = cty - cty_int

                dim_target[batch_id, j] = dim
                depth_target[batch_id, j] = depth[j]
                # multibins
                #import ipdb; ipdb.set_trace()
                loc = gt_bbox_3d[j][0: 3]
                alpha_cls_target[batch_id, j], alpha_offset_target[batch_id, j] = self.angle2class(alpha)
                class_id_target[batch_id, j] = self.angle2class_kl(alpha, loc)
                yaw_target[batch_id, j] = alpha
                loc_target[batch_id, j] = loc
                mask_target[batch_id, j] = 1

                for k in range(self.num_kpt):
                    kpt = gt_kpt_2d_single[k]
                    kptx_int, kpty_int = kpt.int()
                    kptx, kpty = kpt
                    vis_level = gt_kpt_valid_mask_single[k]
                    if vis_level < self.vector_regression_level:
                        continue

                    center2kpt_offset_target[batch_id, j, k * 2] = kptx - ctx_int
                    center2kpt_offset_target[batch_id, j, k * 2 + 1] = kpty - cty_int
                    mask_center2kpt_offset[batch_id, j, k * 2:k * 2 + 2] = 1

                    is_kpt_inside_image = (0 <= kptx_int < feat_w) and (0 <= kpty_int < feat_h)
                    if not is_kpt_inside_image:
                        continue

                    gen_gaussian_target(kpt_heatmap_target[batch_id, k],
                                        [kptx_int, kpty_int], radius)

                    kpt_index = kpty_int * feat_w + kptx_int
                    indices_kpt[batch_id, j, k] = kpt_index

                    kpt_heatmap_offset_target[batch_id, j, k * 2] = kptx - kptx_int
                    kpt_heatmap_offset_target[batch_id, j, k * 2 + 1] = kpty - kpty_int
                    mask_kpt_heatmap_offset[batch_id, j, k * 2:k * 2 + 2] = 1

        indices_kpt = indices_kpt.reshape(bs, -1)
        mask_target = mask_target.type(torch.bool)
        #import ipdb; ipdb.set_trace()
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            center2kpt_offset_target=center2kpt_offset_target,
            dim_target=dim_target,
            depth_target=depth_target,
            alpha_cls_target=alpha_cls_target,
            alpha_offset_target=alpha_offset_target,
            kpt_heatmap_target=kpt_heatmap_target,
            kpt_heatmap_offset_target=kpt_heatmap_offset_target,
            indices=indices,
            indices_kpt=indices_kpt,
            mask_target=mask_target,
            mask_center2kpt_offset=mask_center2kpt_offset,
            mask_kpt_heatmap_offset=mask_kpt_heatmap_offset,
            class_id_target=class_id_target,
            yaw_target=yaw_target,
            loc_target=loc_target,
        )

        return target_result

    @staticmethod
    def extract_input_from_tensor(input, ind, mask):
        input = transpose_and_gather_feat(input, ind)
        return input[mask]

    @staticmethod
    def extract_target_from_tensor(target, mask):
        return target[mask]

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class and residual. '''
        angle = angle % (2 * PI)
        assert (angle >= 0 and angle <= 2 * PI)
        angle_per_class = 2 * PI / float(self.num_alpha_bins)
        shifted_angle = (angle + angle_per_class / 2) % (2 * PI)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def angle2class_kl(self, gt_yaw, loc):
        gt_yaw[gt_yaw > PI] = gt_yaw[gt_yaw > PI] - 2 * PI
        gt_yaw[gt_yaw < -PI] = gt_yaw[gt_yaw < -PI] + 2 * PI
        local_yaw = gt_yaw - torch.atan(loc[0] / (loc[2] + 1e-7))
        local_yaw = local_yaw % (2 * PI)
        angle_per_class = 2 * PI / float(self.num_alpha_bins)
        class_id = local_yaw / angle_per_class
        class_id = int(class_id)
        return class_id

    def class2angle(self, cls, residual):
        ''' Inverse function to angle2class. '''
        angle_per_class = 2 * PI / float(self.num_alpha_bins)
        angle_center = cls * angle_per_class
        angle = angle_center + residual
        return angle

    def decode_alpha_multibin(self, alpha_cls, alpha_offset):
        alpha_score, cls = alpha_cls.max(dim=-1)
        cls = cls.unsqueeze(2)
        alpha_offset = alpha_offset.gather(2, cls)
        alpha = self.class2angle(cls, alpha_offset)

        alpha[alpha > PI] = alpha[alpha > PI] - 2 * PI
        alpha[alpha < -PI] = alpha[alpha < -PI] + 2 * PI
        return alpha

    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   center2kpt_offset_preds,
                   kpt_heatmap_preds,
                   kpt_heatmap_offset_preds,
                   dim_preds,
                   alpha_cls_preds,
                   alpha_offset_preds,
                   depth_preds,
                   alpha_kl_preds,
                   img_metas,
                   rescale=False):

        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) \
               == len(center2kpt_offset_preds) == len(kpt_heatmap_preds) == len(kpt_heatmap_offset_preds) \
               == len(dim_preds) == len(alpha_cls_preds) == len(alpha_offset_preds) == 1
        scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]
        box_type_3d = img_metas[0]['box_type_3d']

        batch_det_bboxes, batch_det_bboxes_3d, batch_labels = self.decode_heatmap(
            center_heatmap_preds[0],
            wh_preds[0],
            offset_preds[0],
            center2kpt_offset_preds[0],
            kpt_heatmap_preds[0],
            kpt_heatmap_offset_preds[0],
            dim_preds[0],
            alpha_cls_preds[0],
            alpha_offset_preds[0],
            depth_preds[0],
            alpha_kl_preds[0],
            img_metas[0]['pad_shape'][:2],
            img_metas[0]['cam_intrinsic'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel,
            thresh=self.test_cfg.thresh)

        if rescale:
            batch_det_bboxes[..., :4] /= batch_det_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        det_results = [
            [box_type_3d(batch_det_bboxes_3d,
                         box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5)),
             batch_det_bboxes[:, -1],
             batch_labels,
             batch_det_bboxes,
             ]
        ]
        return det_results

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       center2kpt_offset_pred,
                       kpt_heatmap_pred,
                       kpt_heatmap_offset_pred,
                       dim_pred,
                       alpha_cls_pred,
                       alpha_offset_pred,
                       depth_pred,
                       alpha_kl_pred,
                       img_shape,
                       camera_intrinsic,
                       k=100,
                       kernel=3,
                       thresh=0.4):
        batch, cat, height, width = center_heatmap_pred.shape
        assert batch == 1
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, ys, xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = xs + offset[..., 0]
        topk_ys = ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)  # (b, k, 5)

        # decode 3D prediction
        dim = transpose_and_gather_feat(dim_pred, batch_index)
        alpha_cls = transpose_and_gather_feat(alpha_cls_pred, batch_index)
        alpha_offset = transpose_and_gather_feat(alpha_offset_pred, batch_index)
        depth_pred = transpose_and_gather_feat(depth_pred, batch_index)
        depth = depth_pred[:, :, 0:1]
        alpha_kl_pred = transpose_and_gather_feat(alpha_kl_pred, batch_index)

        sigma = depth_pred[:, :, 1]
        sigma = torch.exp(-sigma)
        batch_bboxes[..., -1] *= sigma

        center2kpt_offset = transpose_and_gather_feat(center2kpt_offset_pred, batch_index)
        center2kpt_offset = center2kpt_offset.view(batch, k, self.num_kpt * 2)[..., -2:]
        center2kpt_offset[..., ::2] += xs.view(batch, k, 1).expand(batch, k, 1)
        center2kpt_offset[..., 1::2] += ys.view(batch, k, 1).expand(batch, k, 1)

        kpts = center2kpt_offset

        kpts[..., ::2] *= (inp_w / width)
        kpts[..., 1::2] *= (inp_h / height)

        # 1. decode alpha
        alpha = self.decode_alpha_multibin(alpha_cls, alpha_offset)  # (b, k, 1)

        # 1.5 get projected center
        center2d = kpts  # (b, k, 2)
        #import ipdb; ipdb.set_trace()
        # 2. recover rotY
        rot_y = self.recover_rotation(kpts, alpha, camera_intrinsic)  # (b, k, 3)
        # multibins
        # if self.multibins:
        #     rot_y = self.class2angle_kl(kpts, alpha_kl_pred, camera_intrinsic)
        # 2.5 recover box3d_center from center2d and depth
        center3d = torch.cat([center2d, depth], dim=-1).squeeze(0)
        center3d = self.pts2Dto3D(center3d, np.array(camera_intrinsic)).unsqueeze(0)
        #import ipdb; ipdb.set_trace()

        # 3. compose 3D box
        batch_bboxes_3d = torch.cat([center3d, dim, rot_y], dim=-1)

        mask = batch_bboxes[..., -1] > thresh
        batch_bboxes = batch_bboxes[mask]
        batch_bboxes_3d = batch_bboxes_3d[mask]
        batch_topk_labels = batch_topk_labels[mask]

        return batch_bboxes, batch_bboxes_3d, batch_topk_labels

    def recover_rotation(self, kpts, alpha, calib):
        #import ipdb; ipdb.set_trace()
        device = kpts.device
        calib = torch.tensor(calib).type(torch.FloatTensor).to(device).unsqueeze(0)

        si = torch.zeros_like(kpts[:, :, 0:1]) + calib[:, 0:1, 0:1]
        rot_y = alpha + torch.atan2(kpts[:, :, 0:1] - calib[:, 0:1, 2:3], si)

        while (rot_y > PI).any():
            rot_y[rot_y > PI] = rot_y[rot_y > PI] - 2 * PI
        while (rot_y < -PI).any():
            rot_y[rot_y < -PI] = rot_y[rot_y < -PI] + 2 * PI

        return rot_y

    @staticmethod
    def pts2Dto3D(points, view):
        """
        Args:
            points (torch.Tensor): points in 2D images, [N, 3], \
                3 corresponds with x, y in the image and depth.
            view (np.ndarray): camera instrinsic, [3, 3]

        Returns:
            torch.Tensor: points in 3D space. [N, 3], \
                3 corresponds with x, y, z in 3D space.
        """
        assert view.shape[0] <= 4
        assert view.shape[1] <= 4
        assert points.shape[1] == 3

        points2D = points[:, :2]
        depths = points[:, 2].view(-1, 1)
        unnorm_points2D = torch.cat([points2D * depths, depths], dim=1)

        viewpad = torch.eye(4, dtype=points2D.dtype, device=points2D.device)
        viewpad[:view.shape[0], :view.shape[1]] = points2D.new_tensor(view)
        inv_viewpad = torch.inverse(viewpad).transpose(0, 1)

        # Do operation in homogenous coordinates.
        nbr_points = unnorm_points2D.shape[0]
        homo_points2D = torch.cat(
            [unnorm_points2D,
             points2D.new_ones((nbr_points, 1))], dim=1)
        points3D = torch.mm(homo_points2D, inv_viewpad)[:, :3]

        return points3D

    @staticmethod
    def _topk_channel(scores, K=40):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        return topk_scores, topk_inds, topk_ys, topk_xs

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      centers2d=None,
                      depths=None,
                      gt_kpts_2d=None,
                      gt_kpts_valid_mask=None,
                      attr_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x)
        #import ipdb; ipdb.set_trace()
        assert gt_labels is not None
        assert attr_labels is None
        #import ipdb; ipdb.set_trace()
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_bboxes_3d,
                              gt_labels_3d, centers2d, depths, gt_kpts_2d, gt_kpts_valid_mask,
                              img_metas, attr_labels)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        if proposal_cfg is None:
            return losses
        else:
            raise NotImplementedError
