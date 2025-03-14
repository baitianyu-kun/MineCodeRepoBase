import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_util import sample_and_group_multi
from models.pointnet_util import angle
import open3d as o3d


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


def getLocalPPFFeat(radius, n_sample, xyz, normals):
    # Use All points to calculate local ppf features
    features = sample_and_group_multi(-1, radius, n_sample, xyz, normals)
    features = features['ppf']
    return features


def getGlobalPPFFeat(global_xyz, global_nr):
    B, N, C = global_xyz.shape
    d_global = global_xyz.repeat_interleave(
        N, dim=1).view(-1, N, C) - global_xyz.repeat(1, N, 1).view(-1, N, C)
    d_global = d_global.view(B, N, N, C)
    ni_d = angle(global_nr.view(B, N, 1, C), d_global)
    nj_d = angle(global_nr.repeat(1, N, 1).view(B, -1, N, C), d_global)
    ni_nj = angle(global_nr.repeat_interleave(N, dim=1).view(-1, N, C),
                  global_nr.repeat(1, N, 1).view(-1, N, C))
    ni_nj = ni_nj.view(B, N, N)
    d_global_norm = torch.norm(d_global, dim=-1)
    global_ppf_feat = torch.stack(
        [ni_d, nj_d, ni_nj, d_global_norm], dim=-1)  # (B, N, N, 4)
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

        self.prepool_global = get_prepool(4, emb_dims * 2)
        self.postpool_global = get_postpool(emb_dims * 2, emb_dims)

    def embeddings(self, features, prepool, postpool):
        # features: [B, out_dims, neighbors, num_pts]
        # max pooling on neighbours dims...
        features = prepool(features.permute(0, 3, 2, 1))
        features = torch.max(features, 2)[0]
        features = postpool(features)
        features = features.permute(0, 2, 1)
        features = features / torch.norm(features, dim=-1, keepdim=True)
        return features

    def compute_normal(self, pointcloud, radius=0.03, max_nn=30):
        pcd_ = o3d.geometry.PointCloud()
        pcd_.points = o3d.utility.Vector3dVector(pointcloud)
        pcd_.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
        normals = np.asarray(pcd_.normals)
        return normals.astype('float32')

    def forward(self, xyz, normals=None,radius=0.03,max_nn=30):
        # compute normals
        if normals == None:
            normals = self.compute_normal(xyz,radius,max_nn)

        # local ppf feature
        local_ppf_feat = getLocalPPFFeat(
            self.radius, self.n_sample, xyz, normals)
        local_ppf_feat = self.embeddings(
            local_ppf_feat, self.prepool, self.postpool)

        # use global feat
        if self.use_global:
            global_xyz = xyz
            global_nr = normals
            global_ppf_feat = getGlobalPPFFeat(global_xyz, global_nr)
            global_ppf_feat = self.embeddings(
                global_ppf_feat, self.prepool_global, self.postpool_global)
            ppf_feat = local_ppf_feat + global_ppf_feat
        else:
            ppf_feat = local_ppf_feat
        return ppf_feat


if __name__ == '__main__':
    xyz = torch.rand((2, 512, 3))
    normals = torch.rand((2, 512, 3))
    ppf = PPFLocalGlobalNet(emb_dims=96, radius=0.3,
                            num_neighbors=64, use_global=True)
    ppf(xyz, normals)
