# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import os
from typing import Iterable
from logger import setup_logger

import torch

import models.misc as utils
import numpy as np
import torchvision.transforms as standard_transforms
import cv2

logger = setup_logger(__name__)


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def vis(samples, targets, pred, vis_dir, des=None):
    gts = [t['point'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    # draw one by one
    for idx in range(samples.shape[0]):
        sample = restore_transform(samples[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        max_len = np.max(sample_gt.shape)

        size = 2
        # draw gt
        for t in gts[idx]:
            sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
        # draw predictions
        for p in pred[idx]:
            sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        name = targets[idx]['image_id']
        # save the visualized images
        if des is not None:
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_gt.jpg'.format(int(name), 
                                                des, len(gts[idx]), len(pred[idx]))), sample_gt)
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_pred.jpg'.format(int(name), 
                                                des, len(gts[idx]), len(pred[idx]))), sample_pred)
        else:
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_gt.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_gt)
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_pred.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_pred)


# the training routine
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # iterate all training samples
    for img, gt_point, gt_label in data_loader:
        img, gt_point, gt_label = img.to(device), gt_point.to(device), gt_label.to(device)
        outputs = model(img)
        targets = {'points': gt_point, 'labels': gt_label}
        # calc the losses
        losses = criterion(outputs, targets)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        metric_logger.update(loss=losses)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    logger.info(f"Averaged stats: {metric_logger.__str__()}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# the inference routine
@torch.no_grad()
def evaluate_crowd_no_overlap(model, data_loader, device, vis_dir=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []
    for img, count in data_loader:
        img = img.to(device)

        outputs = model(img)
        outputs_points = outputs['pred_points'][0].squeeze(0).view(-1)
        gt_cnt = count

        predict_cnt = outputs_points.sum().detach().cpu().numpy()
        # if specified, save the visualized images
        if vis_dir is not None: 
            vis(samples, targets, [points], vis_dir)
        # accumulate MAE, MSE
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))
    # calc MAE, MSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    return mae, mse
