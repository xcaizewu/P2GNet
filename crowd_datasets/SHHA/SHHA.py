import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import torchvision.transforms.functional as F
import torch.nn.functional as FF
from torchvision import transforms


def gen_discrete_map(im_height, im_width, points):
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map

    # fast create discrete map
    points_np = np.array(points).round().astype(int)
    p_h = np.minimum(points_np[:, 1], np.array([h - 1] * num_gt).astype(
        int))
    p_w = np.minimum(points_np[:, 0], np.array([w - 1] * num_gt).astype(int))
    p_index = torch.from_numpy(p_h * im_width + p_w)
    discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index, src=torch.ones(im_width * im_height)).view(im_height, im_width).numpy()

    assert np.sum(discrete_map) == num_gt
    return discrete_map


class SHHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.train_lists = "part_A_train.txt"
        self.eval_list = "part_A_test.txt"

        if train:
            self.img_list_file = [self.train_lists]
        else:
            self.img_list_file = [self.eval_list]

        self.img_map = {}
        self.img_list = []
        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            with open(os.path.join(self.root_path, train_list)) as fin:
                lines = fin.readlines()
                for line in lines:
                    if len(line) < 2: 
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)
        
        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

        self.c_size = 128
        self.d_ratio = 8
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def train_transform(self, img, keypoints):
        ht = img.shape[1]
        wd = img.shape[2]
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio

        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_labels = (gt_discrete > 0)
        gt_labels = np.expand_dims(gt_labels, 0)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return img, torch.from_numpy(
            gt_discrete.copy()).float(), torch.from_numpy(gt_labels.copy()).float()

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        # load image and ground truth
        img, point = load_data((img_path, gt_path), self.train)

        if self.train:
            width = self.c_size
            height = self.c_size
            img = self.trans(img)
            pad_y = max(0, width - img.shape[1])
            pad_x = max(0, height - img.shape[2])
            if pad_y + pad_x > 0:
                img = FF.pad(img, [0, pad_x, 0, pad_y], value=0)
            img, gt_point, gt_label = self.train_transform(img, point)
            return img, gt_point, gt_label
        else:
            img = self.trans(img)
            return img, len(point)


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # load ground truth points
    points = []
    with open(gt_path) as f_label:
        for line in f_label:
            x = float(line.strip().split(' ')[0])
            y = float(line.strip().split(' ')[1])
            points.append([x, y])
    return img, np.array(points)


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w