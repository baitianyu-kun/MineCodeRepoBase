import numpy as np
import torch
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


class GetOverlapRegion:
    def square_distance(self, src, dst):
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, dim=-1)[:, :, None]
        dist += torch.sum(dst ** 2, dim=-1)[:, None, :]
        return dist

    def get_overlap_region(self, ps, trans_ps, pt, threshold, return_all=False, return_mask=False):
        batch_size, ps_num_pts, _ = trans_ps.shape
        batch_size, pt_num_pts, _ = pt.shape
        dist = self.square_distance(trans_ps, pt)
        dist_judge = torch.where(dist < threshold, True, False)
        pt_overlap_index = torch.any(dist_judge, dim=1, keepdim=True).view(batch_size, pt_num_pts, 1).repeat(1, 1, 3)
        trans_ps_overlap_index = torch.any(dist_judge, dim=2, keepdim=True).repeat(1, 1, 3)
        if return_all:
            # 重叠区域为点值,非重叠区域点的坐标值为0
            ps_overlap_pts = torch.where(trans_ps_overlap_index == True, ps, torch.zeros_like(ps))
            pt_overlap_pts = torch.where(pt_overlap_index == True, pt, torch.zeros_like(pt))
            if return_mask:
                # 重叠区域为1,非重叠区域为0
                # TODO need to change may not incorrect
                ps_overlap_mask = trans_ps_overlap_index.long()
                pt_overlap_mask = pt_overlap_index.long()
                return ps_overlap_mask, pt_overlap_mask
            else:
                return ps_overlap_pts, pt_overlap_pts,ps_overlap_pts_trans
        else:
            ps_overlap_pts = ps[trans_ps_overlap_index].view(batch_size, -1, 3)
            pt_overlap_pts = pt[pt_overlap_index].view(batch_size, -1, 3)
            return ps_overlap_pts, pt_overlap_pts


if __name__ == '__main__':
    # partial_overlap=2
    ps, pt, iGgt, iRgt, itgt, gt_mask_src, gt_mask_tgt = ThreeDMatchPartial(
        DATA_DIR='/mnt/d/dataset/sun3d-home_at-home_at_scan1_2013_jan_1', partition='train',
        num_pts=8000).__getitem__(0)
    import utils.visual as vi

    # Ggt = se3.inverse(torch.from_numpy(iGgt).unsqueeze(0))[0].numpy()
    Ggt = se3.inverse_np(iGgt)
    # vi.show(se3.transform_np(Ggt, ps[gt_mask_src == 1, :]), pt[gt_mask_tgt == 1, :])
    # print(ps.shape)
    # print(ps[gt_mask_src == 1, :].shape)

    # print(pt.shape)
    # print(pt[gt_mask_tgt == 1, :].shape)
    # vi.show(se3.transform_np(Ggt, ps), pt)


    # np.save('ps',ps)
    # np.save('pt',pt)
    # np.save('Ggt',Ggt)
    # ps = np.load('ps.npy')
    # pt = np.load('pt.npy')
    # Ggt = np.load('Ggt.npy')

    ps = torch.from_numpy(ps).unsqueeze(0).float()
    pt = torch.from_numpy(pt).unsqueeze(0).float()
    Ggt = torch.from_numpy(Ggt).unsqueeze(0).float()
    ps = torch.concat([ps, ps])
    pt = torch.concat([pt, pt])
    Ggt = torch.concat([Ggt, Ggt])

    batch_size, num_pts, _ = ps.shape
    trans_ps = se3.transform_torch(Ggt, ps)

    getoverlap = GetOverlapRegion()
    trans_ps_overlap_pts, pt_overlap_pts = getoverlap.get_overlap_region(ps, trans_ps, pt, 0.003, return_all=True,
                                                                         return_mask=False)
    vi.show(trans_ps_overlap_pts[1], pt_overlap_pts[1])
