import torch
import torch.nn as nn
import math

_EPS = 1e-6


def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor = None):
    """ Compute rigid transforms between two point sets
    Args:
        a (torch.Tensor): ([*,] N, 3) points
        b (torch.Tensor): ([*,] N, 3) points
        weights (torch.Tensor): ([*, ] N)
    Returns:
        Transform T ([*,] 4, 4) to get from a to b, i.e. T*a = b
    """
    assert a.shape == b.shape
    assert a.shape[-1] == 3
    if weights is not None:
        assert a.shape[:-1] == weights.shape
        assert weights.min() >= 0 and weights.max() <= 1
        weights_normalized = weights[..., None] / \
                             torch.clamp_min(torch.sum(weights, dim=-1, keepdim=True)[..., None], _EPS)
        centroid_a = torch.sum(a * weights_normalized, dim=-2)
        centroid_b = torch.sum(b * weights_normalized, dim=-2)
        a_centered = a - centroid_a[..., None, :]
        b_centered = b - centroid_b[..., None, :]
        cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)
    else:
        centroid_a = torch.mean(a, dim=-2)
        centroid_b = torch.mean(b, dim=-2)
        a_centered = a - centroid_a[..., None, :]
        b_centered = b - centroid_b[..., None, :]
        cov = a_centered.transpose(-2, -1) @ b_centered
    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[..., 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[..., None, None] > 0, rot_mat_pos, rot_mat_neg)
    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[..., :, None] + centroid_b[..., :, None]
    transform = torch.cat((rot_mat, translation), dim=-1)
    # CHANGED T ([*,] 3, 4) -> T ([*,] 4, 4)
    temp = torch.zeros_like(torch.empty(transform.shape[0], 4, 4))
    temp[:, :3, :4] = transform
    temp[:, 3, 3] = 1
    transform = temp
    return transform


class SVDHead(nn.Module):
    def __init__(self, input_shape="bnc"):
        super(SVDHead, self).__init__()
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.input_shape = input_shape

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size = src.size(0)
        if self.input_shape == "bnc":
            src = src.permute(0, 2, 1)
            tgt = tgt.permute(0, 2, 1)
        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)
        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())
        src_centered = src - src.mean(dim=2, keepdim=True)
        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)
        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())
        U, S, V = [], [], []
        R = []
        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                # r = r * self.reflect
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)
        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)
        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3)
