import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.data_utils import farthest_avg_subsample_points, random_pose, jitter_pcd
from se_math import se3
from utils.datasets import load_data


class ThreeDMatch(Dataset):
    def __init__(self,
                 DATA_DIR,
                 partition,
                 num_pts=1024,
                 max_angle=45, max_t=0.5,
                 noise=0.01,
                 shuffle_pts=True):
        self.data, self.label = load_data(DATA_DIR, partition=partition, file_type='3DMatch_all')
        self.max_angle = max_angle
        self.max_t = max_t
        self.noise = noise
        self.num_pts = num_pts
        self.shuffle_pts = shuffle_pts

    def __getitem__(self, item):
        # template
        pt = self.data[item]
        pt = farthest_avg_subsample_points(pt, self.num_pts)
        Ggt = random_pose(self.max_angle, self.max_t)
        Rgt, tgt = se3.decompose_trans(Ggt)
        # p template -> p source
        # So p source -> inverse Ggt -> p template
        # So model output G: G == inverse Ggt
        # So model output R: R == inverse Rgt
        # So model output t: t == inverse tgt
        ps = se3.transform_np(Ggt, pt)

        if self.noise:
            pt = jitter_pcd(pt, sigma=self.noise, clip=0.05)
            ps = jitter_pcd(ps, sigma=self.noise, clip=0.05)

        if self.shuffle_pts:
            pt = np.random.permutation(pt)
            ps = np.random.permutation(ps)

        return pt, ps, Ggt, Rgt, tgt

    def __len__(self):
        return self.data.shape[0]
