import numpy as np
import torch
import torch.nn as nn
from algo.common import knn
import logging
import torch.nn.functional as F
from RIGARework.ppfnet_util import square_distance, angle_difference

_EPS = 1e-5  # To prevent division by zero


class ParameterPredictionNet(nn.Module):
    def __init__(self, weights_dim):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self.weights_dim = weights_dim
        # Pointnet
        self.prepool = nn.Sequential(
            nn.Conv1d(4, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.postpool = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),

            nn.Linear(256, 2 + np.prod(weights_dim)),
        )
        self._logger.info('Predicting weights with dim {}.'.format(self.weights_dim))

    def forward(self, x):
        src_padded = F.pad(x[0], (0, 1), mode='constant', value=0)
        ref_padded = F.pad(x[1], (0, 1), mode='constant', value=1)
        concatenated = torch.cat([src_padded, ref_padded], dim=1)
        prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
        raw_weights = self.postpool(pooled)
        beta = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])
        return beta, alpha


def get_graph_feature_with_relative_coor(x, k):
    """knn-graph_test.
    Args:
        x: Input point clouds. Size [B, 3, N]
        k: Number of nearest neighbors.
    Returns:
        idx: Nearest neighbor indices. Size [B * N * k]
        relative_coordinates: Relative coordinates between nearest neighbors and the center point. Size [B, 3, N, K]
        knn_points: Coordinates of nearest neighbors. Size[B, N, K, 3].
        idx2: Nearest neighbor indices. Size [B, N, k]
    """
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    idx2 = idx
    batch_size, num_points, _ = idx.size()
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    knn_points = x.view(batch_size * num_points, -1)[idx, :]
    knn_points = knn_points.view(batch_size, num_points, k, num_dims)  # [b, n, k, 3],knn
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # [b, n, k, 3],central points
    relative_coordinates = (knn_points - x).permute(0, 3, 1, 2)
    return idx, relative_coordinates, knn_points, idx2


def match_features(feat_src, feat_ref, metric='l2'):
    assert feat_src.shape[-1] == feat_ref.shape[-1]
    if metric == 'l2':
        dist_matrix = square_distance(feat_src, feat_ref)
    elif metric == 'angle':
        feat_src_norm = feat_src / (torch.norm(feat_src, dim=-1, keepdim=True) + _EPS)
        feat_ref_norm = feat_ref / (torch.norm(feat_ref, dim=-1, keepdim=True) + _EPS)
        dist_matrix = angle_difference(feat_src_norm, feat_ref_norm)
    else:
        raise NotImplementedError
    return dist_matrix
