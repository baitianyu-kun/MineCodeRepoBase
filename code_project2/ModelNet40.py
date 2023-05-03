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


class ModelNet40(Dataset):
    def __init__(self,
                 DATA_DIR,
                 partition,
                 partial_overlap=0,
                 num_pts=1024,
                 max_angle=45, max_t=0.5,
                 subsampled_rate_src=0.7,
                 subsampled_rate_tgt=0.7):
        logger = logging.getLogger()
        logger.info(f'Start loading {partition} data.....')

        self.data, self.label = load_data(DATA_DIR, partition=partition, file_type='modelnet40')
        self.max_angle = max_angle
        self.max_t = max_t
        self.num_pts = num_pts
        self.partial_overlap = partial_overlap
        self.subsampled_rate_src = subsampled_rate_src
        self.subsampled_rate_tgt = subsampled_rate_tgt

    def __getitem__(self, item):
        points = self.data[item][:self.num_pts]
        iGgt = random_pose(self.max_angle, self.max_t)
        iRgt, itgt = se3.decompose_trans(iGgt)
        if self.partial_overlap == 2:
            src_subsampled_points = int(self.subsampled_rate_src * points.shape[0])
            tgt_subsampled_points = int(self.subsampled_rate_tgt * points.shape[0])
            pointcloud1, mask_src, pointcloud2, mask_tgt = farthest_neighbour_subsample_points2(
                points, src_subsampled_points, tgt_subsampled_points)
            gt_mask_src, gt_mask_tgt = get_src_tgt_mask(mask_src, mask_tgt)

            pt = pointcloud2
            ps = se3.transform_np(iGgt, pointcloud1)
            return ps.astype('float32'), pt.astype('float32'), \
                   iGgt.astype('float32'), iRgt.astype('float32'), itgt.astype('float32'), \
                   gt_mask_src, gt_mask_tgt

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    ps, pt, iGgt, iRgt, itgt, gt_mask_src, gt_mask_tgt = ModelNet40(DATA_DIR='D:\\dataset',
                                                                    partition='train',
                                                                    partial_overlap=2).__getitem__(500)
    # import cv2 as cv
    # cv.imshow('test',match_mtx.detach().cpu().numpy())
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # import utils.visual as vi
    #
    # Ggt = se3.inverse_np(iGgt)
    # vi.show(se3.transform_np(Ggt, ps[gt_mask_src == 1, :]), pt[gt_mask_tgt == 1, :])
    # vi.show(se3.transform_np(Ggt, ps), pt)
