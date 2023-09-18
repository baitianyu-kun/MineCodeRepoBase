import numpy as np
from torch.utils.data import Dataset
from utils.data_utils import farthest_avg_subsample_points, random_pose, jitter_pcd, \
    farthest_neighbour_subsample_points2, get_src_tgt_mask
from se_math import se3
from utils.datasets import load_data
import logging


class ThreeDMatchPartial(Dataset):
    def __init__(self,
                 DATA_DIR,
                 partition,
                 partial_overlap=2,
                 num_pts=1024,
                 max_angle=45, max_t=0.5,
                 noise=0.0,
                 shuffle_pts=True,
                 subsampled_rate_src=0.6,
                 subsampled_rate_tgt=0.6):

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
        iGgt = random_pose(self.max_angle, self.max_t)
        iRgt, itgt = se3.decompose_trans(iGgt)
        if self.partial_overlap == 2:
            src_subsampled_points = int(self.subsampled_rate_src * points.shape[0])
            tgt_subsampled_points = int(self.subsampled_rate_tgt * points.shape[0])
            # mask_src, mask_tgt都是相对于原始points的mask,所以下面需要得到src和tgt之间的mask
            # pointcloud1,pointcloud2其实都是一样的,就是不同的重叠部分罢了,都可以认为是template
            pointcloud1, mask_src, pointcloud2, mask_tgt = farthest_neighbour_subsample_points2(
                points, src_subsampled_points, tgt_subsampled_points)
            gt_mask_src, gt_mask_tgt = get_src_tgt_mask(mask_src, mask_tgt)
            pt = pointcloud2
            ps = se3.transform_np(iGgt, pointcloud1)
            if self.noise:
                ps = jitter_pcd(ps, sigma=self.noise, clip=0.05)
            if self.shuffle_pts:
                state = np.random.get_state()
                ps = np.random.permutation(ps)
                np.random.set_state(state)
                gt_mask_src = np.random.permutation(gt_mask_src)
            return ps.astype('float32'), pt.astype('float32'), \
                   iGgt.astype('float32'), iRgt.astype('float32'), itgt.astype('float32'), \
                   gt_mask_src, gt_mask_tgt
        elif self.partial_overlap == 1:
            pt = points
            ps = se3.transform_np(iGgt, pt)
            subsampled_points = int(self.subsampled_rate_tgt * points.shape[0])
            ps, gt_mask_src = farthest_neighbour_subsample_points2(ps, subsampled_points)
            if self.noise:
                ps = jitter_pcd(ps, sigma=self.noise, clip=0.05)
            if self.shuffle_pts:
                state = np.random.get_state()
                pt = np.random.permutation(pt)
                np.random.set_state(state)
                gt_mask_src = np.random.permutation(gt_mask_src)
            return ps.astype('float32'), pt.astype('float32'), iGgt.astype('float32'), iRgt.astype(
                'float32'), itgt.astype('float32'), gt_mask_src
        elif self.partial_overlap == 0:
            pt = points
            ps = se3.transform_np(iGgt, pt)
            if self.noise:
                ps = jitter_pcd(ps, sigma=self.noise, clip=0.05)
            if self.shuffle_pts:
                ps = np.random.permutation(ps)
            return ps.astype('float32'), pt.astype('float32'), \
                   iGgt.astype('float32'), iRgt.astype('float32'), itgt.astype('float32')

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    # partial_overlap=2
    ps, pt, iGgt, iRgt, itgt, gt_mask_src, gt_mask_tgt = ThreeDMatchPartial(
        DATA_DIR='D:\\dataset\\sun3d-home_at-home_at_scan1_2013_jan_1', partition='train',
        num_pts=9000).__getitem__(0)
    import utils.visual as vi

    # Ggt = se3.inverse(torch.from_numpy(iGgt).unsqueeze(0))[0].numpy()
    Ggt = se3.inverse_np(iGgt)
    vi.show(se3.transform_np(Ggt, ps[gt_mask_src == 1, :]), pt[gt_mask_tgt == 1, :])
    print(ps.shape)
    print(ps[gt_mask_src == 1, :].shape)

    print(pt.shape)
    print(pt[gt_mask_tgt == 1, :].shape)
    vi.show(se3.transform_np(Ggt, ps), pt+0.02)
