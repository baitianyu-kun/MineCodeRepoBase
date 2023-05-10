import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from RIGARework.ppfnet_util import sample_and_group_multi
from RIGARework.ppfnet_util import angle

_raw_features_sizes = {'xyz': 3, 'dxyz': 3, 'ppf': 4}
_raw_features_order = {'xyz': 0, 'dxyz': 1, 'ppf': 2}


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


class PPFNet(nn.Module):
    """Feature extraction Module that extracts hybrid features"""

    def __init__(self, features=['ppf', 'dxyz', 'xyz'], emb_dims=96, radius=0.3, num_neighbors=64, use_global=False):
        super().__init__()

        self.use_global=use_global

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info('Using early fusion, feature dim = {}'.format(emb_dims))
        self.radius = radius
        self.n_sample = num_neighbors

        self.features = sorted(features, key=lambda f: _raw_features_order[f])
        self._logger.info('Feature extraction using features {}'.format(', '.join(self.features)))

        # Layers
        raw_dim = np.sum([_raw_features_sizes[f] for f in self.features])  # number of channels after concat
        self.prepool = get_prepool(raw_dim, emb_dims * 2)
        self.postpool = get_postpool(emb_dims * 2, emb_dims)

        self.prepool_global = get_prepool(4, emb_dims * 2)
        self.postpool_global = get_postpool(emb_dims * 2, emb_dims)

    def forward(self, xyz, normals):
        features = sample_and_group_multi(-1, self.radius, self.n_sample, xyz, normals)
        features['xyz'] = features['xyz'][:, :, None, :]

        # Gate and concat
        concat = []
        for i in range(len(self.features)):
            f = self.features[i]
            expanded = (features[f]).expand(-1, -1, self.n_sample, -1)
            concat.append(expanded)
        fused_input_feat = torch.cat(concat, -1)

        # Prepool_FC, pool, postpool-FC
        new_feat = fused_input_feat.permute(0, 3, 2, 1)  # [B, 10, n_sample, N]
        new_feat = self.prepool(new_feat)

        pooled_feat = torch.max(new_feat, 2)[0]  # Max pooling (B, C, N)

        post_feat = self.postpool(pooled_feat)  # Post pooling dense layers
        cluster_feat = post_feat.permute(0, 2, 1)
        cluster_feat = cluster_feat / torch.norm(cluster_feat, dim=-1, keepdim=True)

        # begin global feat
        if self.use_global:
            global_xyz = xyz
            global_nr = normals
            B, N, C = global_xyz.shape
            d_global = global_xyz.repeat_interleave(N, dim=1).view(-1, N, C) - global_xyz.repeat(1, N, 1).view(-1, N, C)
            d_global = d_global.view(B, N, N, C)
            ni_d = angle(global_nr.view(B, N, 1, C), d_global)
            nj_d = angle(global_nr.repeat(1, N, 1).view(B, -1, N, C), d_global)
            ni_nj = angle(global_nr.repeat_interleave(N, dim=1).view(-1, N, C), global_nr.repeat(1, N, 1).view(-1, N, C))
            ni_nj = ni_nj.view(B, N, N)
            d_global_norm = torch.norm(d_global, dim=-1)
            ppf_global_feat = torch.stack([ni_d, nj_d, ni_nj, d_global_norm], dim=-1)  # (B, N, N, 4)
            ppf_global_feat = self.prepool_global(ppf_global_feat.permute(0, 3, 2, 1))
            ppf_global_feat = torch.max(ppf_global_feat, 2)[0]
            ppf_global_feat = self.postpool_global(ppf_global_feat)
            ppf_global_feat = ppf_global_feat.permute(0, 2, 1)
            ppf_global_feat = ppf_global_feat / torch.norm(ppf_global_feat, dim=-1, keepdim=True)

            cluster_feat = cluster_feat + ppf_global_feat
        return cluster_feat  # (B, N, C)


if __name__ == '__main__':
    xyz = torch.rand((2, 1024, 3))
    normals = torch.rand((2, 1024, 3))
    ppf = PPFNet(features=['ppf'], emb_dims=96, radius=0.3, num_neighbors=64)
    print(ppf)
    ppf(xyz, normals)
