import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import visual, cors_utils
from RIGARework.ppfnet_util import *


def get_prepool(in_dim, out_dim):
    """Shared FC part in PointNet before max pooling"""
    net = nn.Sequential(
        nn.Conv2d(in_dim, out_dim // 2, 1),
        nn.GroupNorm(8, out_dim // 2),
        nn.ReLU(),
        nn.Conv2d(out_dim // 2, out_dim // 2, 1),
        nn.GroupNorm(8, out_dim // 2),
        nn.ReLU(),
        nn.Conv2d(out_dim // 2, out_dim, 1),
        nn.GroupNorm(8, out_dim),
        nn.ReLU(),
    )
    return net


def get_postpool(in_dim, out_dim):
    """Linear layers in PointNet after max pooling

    Args:
        in_dim: Number of input channels
        out_dim: Number of output channels. Typically smaller than in_dim

    """
    net = nn.Sequential(
        nn.Conv1d(in_dim, in_dim, 1),
        nn.GroupNorm(8, in_dim),
        nn.ReLU(),
        nn.Conv1d(in_dim, out_dim, 1),
        nn.GroupNorm(8, out_dim),
        nn.ReLU(),
        nn.Conv1d(out_dim, out_dim, 1),
    )

    return net


def getPPFGlobalFeat(global_xyz, global_nr, B, S, C):
    # tensor([[[ 0.7442, -0.1349,  0.2600]]])
    # new_xyz.repeat_interleave(2,dim=1)
    # tensor([[[ 0.7442, -0.1349,  0.2600],
    #          [ 0.7442, -0.1349,  0.2600]]])
    # new_xyz_repeat的第一行减去new_xyz的第一行,new_xyz_repeat的第一行减去new_xyz的第二行........第N减去第N行
    # 得到了每两个点之间的向量,自己和自己之间的向量为0,多batch不能用广播机制
    d = global_xyz.repeat_interleave(S, dim=1).view(-1, S, C) - global_xyz.repeat(1, S, 1).view(-1, S, C)
    # d: [B, -1, C]
    d = d.view(B, S, S, C)
    # ni_d: [B, S, S] 当前normals和d的angle
    ni_d = angle(global_nr.view(B, S, 1, C), d)
    # nj_d: [B, S, S] 其他normals(包括当前normals)和d的angle
    nj_d = angle(global_nr.repeat(1, S, 1).view(B, -1, S, C), d)
    ni_nj = angle(global_nr.repeat_interleave(S, dim=1).view(-1, S, C), global_nr.repeat(1, S, 1).view(-1, S, C))
    # ni_nj: [B, S, S]
    ni_nj = ni_nj.view(B, S, S)
    d_norm = torch.norm(d, dim=-1)

    ppf_global_feat = torch.stack([ni_d, nj_d, ni_nj, d_norm], dim=-1)  # (B, S, S, 4)
    return ppf_global_feat


def getPPFLocalFeat(xyz, normals, new_xyz, nr, fps_idx, S, radius, num_neighbors):
    B, N, C = xyz.shape

    # 用new_xyz去查询xyz的周围的neighbours
    # (B, farthest_npoint, num_neighbors)
    idx = query_ball_point(radius, num_neighbors, xyz, new_xyz, fps_idx)
    # (B, farthest_npoint, num_neighbors, C)
    grouped_xyz = index_points(xyz, idx)
    # d = p_r - p_i  (B, farthest_npoint, num_neighbors , C)
    d = grouped_xyz - new_xyz.view(B, S, 1, C)

    # ni是support normals
    # ni: [B, farthest_npoint, num_neighbors, C]
    ni = index_points(normals, idx)
    # nr是最远点采样后的normals
    # [B, farthest_npoint, 1, C]
    nr = nr[:, :, None, :]

    # nr_d: [B, farthest_npoint, num_neighbors]
    # ni_d: [B, farthest_npoint, num_neighbors]
    # nr_ni: [B, farthest_npoint, num_neighbors]
    nr_d = angle(nr, d)
    ni_d = angle(ni, d)
    nr_ni = angle(nr, ni)

    d_norm = torch.norm(d, dim=-1)

    ppf_local_feat = torch.stack([nr_d, ni_d, nr_ni, d_norm], dim=-1)  # (B, farthest_npoint, n_sample, 4)
    return ppf_local_feat,d


class PPFNetLocalGlobal(nn.Module):

    def __init__(self, emb_dims=96, radius=0.3, num_neighbors=64, farthest_subsample_numpts=512):
        super().__init__()
        self.radius = radius
        # 最大的neighbours采样数
        self.num_neighbors = num_neighbors
        # 最远点采样的点数,用来在计算local ppf feature
        self.farthest_subsample_numpts = farthest_subsample_numpts
        self.emb_dims = emb_dims
        # dim: ppf: 4, points: 3
        self.prepool = get_prepool(4, emb_dims * 2)
        self.postpool = get_postpool(emb_dims * 2, emb_dims)

        self.prepool_global = get_prepool(7, emb_dims * 2)
        self.postpool_global = get_postpool(emb_dims * 2, emb_dims)

    def forward(self, xyz, normals):
        B, N, C = xyz.shape
        S = self.farthest_subsample_numpts
        # 进行最远点采样
        fps_idx = farthest_point_sample(xyz, S)  # [B, farthest_npoint, C]
        # new_xyz是最远点采样后的xyz
        new_xyz = index_points(xyz, fps_idx)
        # nr是最远点采样后的normals
        nr = index_points(normals, fps_idx)

        # [B, farthest_npoint, num_neighbors, 4]
        ppf_local_feat,d = getPPFLocalFeat(xyz, normals, new_xyz, nr, fps_idx,
                                         S, self.radius, self.num_neighbors)
        # [B, farthest_npoint, farthest_npoint, 4+C]
        # ppf_local_feat = torch.cat([ppf_local_feat, new_xyz[:, :, None, :].repeat(1, 1, self.num_neighbors, 1)], -1)
        # ppf_local_feat=torch.cat([ppf_local_feat,d],-1)

        # [B, farthest_npoint, farthest_npoint, 4]
        ppf_global_feat = getPPFGlobalFeat(new_xyz, nr, B, S, C)
        # [B, farthest_npoint, farthest_npoint, 4+C]
        ppf_global_feat = torch.cat([ppf_global_feat, new_xyz[:, :, None, :].repeat(1, 1, S, 1)], -1)

        ppf_local_feat = self.prepool(ppf_local_feat.permute(0, 3, 2, 1))
        ppf_local_feat = torch.max(ppf_local_feat, 2)[0]
        ppf_local_feat = self.postpool(ppf_local_feat)
        ppf_local_feat = ppf_local_feat.permute(0, 2, 1)
        # [B, S, emb_dims] = [1, 5, 96]
        ppf_local_feat = ppf_local_feat / torch.norm(ppf_local_feat, dim=-1, keepdim=True)

        ppf_global_feat = self.prepool_global(ppf_global_feat.permute(0, 3, 2, 1))
        # CHANGED torch.max(ppf_global_feat, 3) 注意dim值
        ppf_global_feat = torch.max(ppf_global_feat, 3)[0]
        ppf_global_feat = self.postpool_global(ppf_global_feat)
        ppf_global_feat = ppf_global_feat.permute(0, 2, 1)
        # [B, S, emb_dims] = [1, 5, 96]
        ppf_global_feat = ppf_global_feat / torch.norm(ppf_global_feat, dim=-1, keepdim=True)

        ppf_feat = ppf_local_feat + ppf_global_feat

        return ppf_feat, new_xyz, nr


if __name__ == '__main__':
    data = np.loadtxt('../../test/data/airplane_0627.txt', delimiter=',')
    xyz = torch.from_numpy(data[:1024, 0:3].astype('float32')).unsqueeze(0)
    normals = torch.from_numpy(data[:1024, 3:6].astype('float32')).unsqueeze(0)
    ppf = PPFNetLocalGlobal()
    ppf(torch.rand((2, 1024, 3)), torch.rand((2, 1024, 3)))
