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
                 subsampled_rate_src=0.8,
                 subsampled_rate_tgt=0.8,
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

    def own_data(self):
        points = np.loadtxt('airplane_0627.txt', delimiter=',')[:1024, :3]
        return points

    def __getitem__(self, item):
        # points = self.data[item][:self.num_pts]
        points = self.own_data()
        iGgt = random_pose(self.max_angle, self.max_t)
        iRgt, itgt = se3.decompose_trans(iGgt)
        if self.partial_overlap == 2:
            src_subsampled_points = int(
                self.subsampled_rate_src * points.shape[0])
            tgt_subsampled_points = int(
                self.subsampled_rate_tgt * points.shape[0])
            pointcloud1, mask_src, pointcloud2, mask_tgt = farthest_neighbour_subsample_points2(
                points, src_subsampled_points, tgt_subsampled_points)
            gt_mask_src, gt_mask_tgt = get_src_tgt_mask(mask_src, mask_tgt)
            pt = pointcloud2
            ps = se3.transform_np(iGgt, pointcloud1)
            return points.astype('float32'), ps.astype('float32'), pt.astype('float32'), iGgt.astype(
                'float32'), gt_mask_src, gt_mask_tgt, mask_src, mask_tgt

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    points, ps, pt, iGgt, gt_mask_src, gt_mask_tgt, mask_src, mask_tgt = ModelNet40Overlap(
        DATA_DIR='D:\\dataset\\', partition='test',
        num_pts=1024).__getitem__(0)
    Ggt = se3.inverse_np(iGgt)
    import utils.visual as vi

    # np.save('ps.npy',ps)
    # np.save('pt.npy',pt)
    # np.save('gt_mask_src.npy',gt_mask_src)
    # np.save('gt_mask_tgt.npy',gt_mask_tgt)
    # np.save('iGgt.npy',iGgt)
    np.save('mask_src.npy',mask_src)
    np.save('mask_tgt.npy',mask_tgt)


    # vi.show(se3.transform_np(Ggt, ps[gt_mask_src == 1, :]), pt[gt_mask_tgt == 1, :])
    # vi.show(se3.transform_np(Ggt, ps), pt)
