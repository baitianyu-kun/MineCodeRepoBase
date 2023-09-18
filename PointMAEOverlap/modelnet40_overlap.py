import copy
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.data_utils import farthest_avg_subsample_points, random_pose, jitter_pcd, \
    farthest_neighbour_subsample_points2, get_src_tgt_mask
from se_math import se3
from utils.datasets import load_data
import logging


class ModelNet40Overlap(Dataset):
    def __init__(self,
                 DATA_DIR,
                 partition,
                 partial_overlap=2,
                 num_pts=1024,
                 subsampled_rate_src=0.7,
                 subsampled_rate_tgt=0.7,
                 noise=0.01, shuffle_pts=True, max_angle=45, max_t=0.5, ):
        logger = logging.getLogger()
        logger.info(f'Start loading {partition} data.....')
        self.data, self.label = load_data(
            DATA_DIR, partition=partition, file_type='modelnet40')
        self.num_pts = num_pts
        self.partial_overlap = partial_overlap
        self.subsampled_rate_src = subsampled_rate_src
        self.subsampled_rate_tgt = subsampled_rate_tgt
        self.noise = noise
        self.shuffle_pts = shuffle_pts
        self.max_angle = max_angle
        self.max_t = max_t

    def __getitem__(self, item):
        points = self.data[item][:self.num_pts]
        iGgt = random_pose(self.max_angle, self.max_t)
        iRgt, itgt = se3.decompose_trans(iGgt)
        if self.partial_overlap == 2:
            src_subsampled_points = int(
                self.subsampled_rate_src * points.shape[0])
            tgt_subsampled_points = int(
                self.subsampled_rate_tgt * points.shape[0])
            pointcloud1, mask_p1, pointcloud2, mask_p2 = farthest_neighbour_subsample_points2(
                points, src_subsampled_points, tgt_subsampled_points)
            pt = pointcloud2
            ps = se3.transform_np(iGgt, pointcloud1)
            if self.noise:
                ps = jitter_pcd(ps, sigma=self.noise, clip=0.05)
            if self.shuffle_pts:
                ps = np.random.permutation(ps)
            return ps.astype('float32'), pt.astype('float32'), iGgt.astype('float32')

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    ps, pt, iGgt = ModelNet40Overlap(
        DATA_DIR='D:\\dataset\\', partition='train',
        num_pts=1024).__getitem__(0)
    Ggt = se3.inverse_np(iGgt)
    import utils.visual as vi

    vi.show(se3.transform_np(Ggt, ps), pt)
