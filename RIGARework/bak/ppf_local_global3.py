import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from RIGARework.ppfnet_util import sample_and_group_multi
from RIGARework.ppfnet_util import angle


def get_prepool(in_dim, out_dim):
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


def getLocalPPFFeat(radius, n_sample,xyz,normals):
    features = sample_and_group_multi(-1, radius, n_sample, xyz, normals)
    features = features['ppf']
    return features


def getGlobalPPFFeat(global_xyz, global_nr):
    B, N, C = global_xyz.shape
    d_global = global_xyz.repeat_interleave(N, dim=1).view(-1, N, C) - global_xyz.repeat(1, N, 1).view(-1, N, C)
    d_global = d_global.view(B, N, N, C)
    ni_d = angle(global_nr.view(B, N, 1, C), d_global)
    nj_d = angle(global_nr.repeat(1, N, 1).view(B, -1, N, C), d_global)
    ni_nj = angle(global_nr.repeat_interleave(N, dim=1).view(-1, N, C),
                  global_nr.repeat(1, N, 1).view(-1, N, C))
    ni_nj = ni_nj.view(B, N, N)
    d_global_norm = torch.norm(d_global, dim=-1)
    global_ppf_feat = torch.stack([ni_d, nj_d, ni_nj, d_global_norm], dim=-1)  # (B, N, N, 4)
    return global_ppf_feat


class PPFLocalGlobalNet(nn.Module):

    def __init__(self, emb_dims=96, radius=0.3, num_neighbors=64, use_global=False):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self.radius = radius
        self.n_sample = num_neighbors
        self.use_global = use_global

        self._logger.info('Feature dim = {}'.format(emb_dims))
        self._logger.info('Using global PPF Feature = {}'.format(use_global))

        self.prepool = get_prepool(4, emb_dims * 2)
        self.postpool = get_postpool(emb_dims * 2, emb_dims)

    def forward(self, xyz, normals):
        # local ppf feature
        local_ppf_feat = getLocalPPFFeat(self.radius, self.n_sample,xyz,normals)
        local_ppf_feat = self.prepool(local_ppf_feat.permute(0, 3, 2, 1))
        local_ppf_feat = torch.max(local_ppf_feat, 2)[0]  # Max pooling (B, C, N)
        local_ppf_feat = self.postpool(local_ppf_feat)  # Post pooling dense layers
        local_ppf_feat = local_ppf_feat.permute(0, 2, 1)
        local_ppf_feat = local_ppf_feat / torch.norm(local_ppf_feat, dim=-1, keepdim=True)

        # begin global feat
        if self.use_global:
            global_xyz = xyz
            global_nr = normals
            global_ppf_feat = getGlobalPPFFeat(global_xyz, global_nr)
            global_ppf_feat = self.prepool(global_ppf_feat.permute(0, 3, 2, 1))
            global_ppf_feat = torch.max(global_ppf_feat, 2)[0]
            global_ppf_feat = self.postpool(global_ppf_feat)
            global_ppf_feat = global_ppf_feat.permute(0, 2, 1)
            global_ppf_feat = global_ppf_feat / torch.norm(global_ppf_feat, dim=-1, keepdim=True)
            ppf_feat = local_ppf_feat + global_ppf_feat
        else:
            ppf_feat = local_ppf_feat
        return ppf_feat


if __name__ == '__main__':
    xyz = torch.rand((2, 1024, 3))
    normals = torch.rand((2, 1024, 3))
    ppf = PPFLocalGlobalNet(emb_dims=96, radius=0.3, num_neighbors=64, use_global=True)
    print(ppf)
    ppf(xyz, normals)
