import copy
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import utils.visual
from utils.data_utils import farthest_avg_subsample_points, random_pose, jitter_pcd, \
    farthest_neighbour_subsample_points2, get_src_tgt_mask
from se_math import se3
from utils.datasets import load_data
import logging
import open3d as o3d


def compute_normal(pointcloud, radius=0.03, max_nn=30):
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(pointcloud)
    pcd_.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    normals = np.asarray(pcd_.normals)
    return normals


class ThreeDMatchRIGA(Dataset):
    def __init__(self,
                 DATA_DIR,
                 partition,
                 partial_overlap=0,
                 num_pts=1024,
                 max_angle=45, max_t=0.5,
                 noise=0.0,
                 shuffle_pts=True,
                 subsampled_rate_src=0.7,
                 subsampled_rate_tgt=0.7):

        logger = logging.getLogger()
        logger.info(f'Start loading {partition} data.....')

        self.data, self.label = load_data(DATA_DIR, partition=partition, file_type='3DMatchHome')
        self.max_angle = max_angle
        self.max_t = max_t
        self.noise = noise
        self.num_pts = num_pts
        self.shuffle_pts = shuffle_pts
        self.partial_overlap = partial_overlap
        self.subsampled_rate_src = subsampled_rate_src
        self.subsampled_rate_tgt = subsampled_rate_tgt

    def __getitem__(self, item):
        points = self.data[item][:self.num_pts]
        normals = compute_normal(points, radius=0.03, max_nn=30)
        iGgt = random_pose(self.max_angle, self.max_t)
        iRgt, itgt = se3.decompose_trans(iGgt)
        if self.partial_overlap == 0:
            pt = points
            nt = normals
            ps = se3.transform_np(iGgt, pt)
            ns = se3.transform_np(iGgt, nt)
            if self.noise:
                ps = jitter_pcd(ps, sigma=self.noise, clip=0.05)
            if self.shuffle_pts:
                state = np.random.get_state()
                ps = np.random.permutation(ps)
                np.random.set_state(state)
                ns = np.random.permutation(ns)
            return ps.astype('float32'), pt.astype('float32'), ns.astype('float32'), nt.astype('float32'), \
                   iGgt.astype('float32'), iRgt.astype('float32'), itgt.astype('float32')

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    pass

    # # partial_overlap=0
    # ps, pt, ns, nt, ig, ir, it = ThreeDMatch(DATA_DIR='D:\\dataset\\sun3d-home_at-home_at_scan1_2013_jan_1',
    #                                          partition='train',
    #                                          partial_overlap=0, num_pts=8000).__getitem__(0)
    # utils.visual.show(ps, pt, ns, nt)
