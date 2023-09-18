import torch
import torch.nn as nn
from RIGARework.LearnOverlapBase import PointNet, Decoder
from RIGARework.CrossAttention import CrossAttention
from utils.data_utils import non_random_farthest_avg_subsample_points
from RIGARework.partial_threeDmatchDataset import ThreeDMatchPartial


class LearnOverlapRegion(nn.Module):

    def __init__(self, partial_num_points=408):
        super(LearnOverlapRegion, self).__init__()
        self.partial_num_points = partial_num_points
        self.encoder = PointNet(dim_k=1024)
        self.decoder = Decoder(num_points=self.partial_num_points, bottleneck_size=1024)
        self.self_attention = CrossAttention(feature_dim=1024, num_heads=1)
        self.cross_attention = CrossAttention(feature_dim=1024, num_heads=1)

    def forward(self, ps, pt):
        # fs: b, dims, n, fs_maxpool: b, n
        fs, fs_maxpool = self.encoder(ps)
        fs_self = fs + self.self_attention(fs, fs)
        ft, ft_maxpool = self.encoder(pt)
        ft_self = ft + self.self_attention(ft, ft)

        fs_cross = fs_self + self.cross_attention(fs_self, ft_self)
        ft_cross = ft_self + self.cross_attention(ft_self, fs_self)

        fs_final = torch.max(fs_cross, dim=2)[0]
        ft_final = torch.max(ft_cross, dim=2)[0]

        # decode overlap region
        decode_ps_overlap = self.decoder(fs_final)
        decode_pt_overlap = self.decoder(ft_final)

        return decode_ps_overlap, decode_pt_overlap


if __name__ == '__main__':
    ps, pt, iGgt, iRgt, itgt, gt_mask_src, gt_mask_tgt = ThreeDMatchPartial(
        DATA_DIR='D:\\dataset\\sun3d-home_at-home_at_scan1_2013_jan_1', partition='test',
        num_pts=1024).__getitem__(0)

    ps_overlap_region = ps[gt_mask_src == 1, :].copy()
    pt_overlap_region = pt[gt_mask_tgt == 1, :].copy()

    ps_overlap_region_fps = non_random_farthest_avg_subsample_points(ps_overlap_region, npoint=512)
    pt_overlap_region_fps = non_random_farthest_avg_subsample_points(pt_overlap_region, npoint=512)

    ps_overlap_region = ps_overlap_region.astype('float32')
    pt_overlap_region = pt_overlap_region.astype('float32')
    ps_overlap_region_fps = ps_overlap_region_fps.astype('float32')
    pt_overlap_region_fps = pt_overlap_region_fps.astype('float32')

    ps = torch.from_numpy(ps).unsqueeze(0).repeat(2, 1, 1)
    pt = torch.from_numpy(pt).unsqueeze(0).repeat(2, 1, 1)
    ps_overlap_region = torch.from_numpy(ps_overlap_region).unsqueeze(0).repeat(2, 1, 1)
    pt_overlap_region = torch.from_numpy(pt_overlap_region).unsqueeze(0).repeat(2, 1, 1)
    ps_overlap_region_fps = torch.from_numpy(ps_overlap_region_fps).unsqueeze(0).repeat(2, 1, 1)
    pt_overlap_region_fps = torch.from_numpy(pt_overlap_region_fps).unsqueeze(0).repeat(2, 1, 1)

    learn = LearnOverlapRegion(partial_num_points=ps_overlap_region.shape[1])

    learn(ps, pt)
