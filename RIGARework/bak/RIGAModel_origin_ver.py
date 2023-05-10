import torch
import torch.nn as nn
import numpy as np
from algo.sinkhorn import sinkhorn
import se_math.se3 as se3
import utils.data_utils
from RIGARework.RIGA_base import match_features
from algo.svd import compute_rigid_transform
from RIGARework.bak.ppf_local_global_origin_ver import PPFNetLocalGlobal


class AttentionBlock(nn.Module):
    def __init__(self, in_dims=96, num_heads=1, drop_out=0.0):
        super(AttentionBlock, self).__init__()
        self.projection_q = nn.Conv1d(in_dims, in_dims, kernel_size=1)
        self.projection_k = nn.Conv1d(in_dims, in_dims, kernel_size=1)
        self.projection_v = nn.Conv1d(in_dims, in_dims, kernel_size=1)
        # heads should be divided by in_dims
        self.feat_attention = nn.MultiheadAttention(embed_dim=in_dims, num_heads=num_heads,
                                                    dropout=drop_out)

        self.multihead_message = nn.Conv1d(in_dims, in_dims, kernel_size=1)
        self.final_message = nn.Sequential(
            nn.Conv1d(in_dims * 2, in_dims * 2, kernel_size=1),
            nn.InstanceNorm1d(in_dims * 2),
            nn.ReLU(),
            nn.Conv1d(in_dims * 2, in_dims, kernel_size=1),
            nn.InstanceNorm1d(in_dims),
            nn.ReLU()
        )
        self.num_heads = num_heads
        self.drop_out = drop_out

    def forward(self, feature, opposite_feature=None):
        feature = feature.transpose(1, 2)
        batch, dims, num_pts = feature.shape
        Q = self.projection_q(feature)
        K = self.projection_k(feature)
        V = self.projection_v(feature)

        # [B, in_dims, num_pts]
        attn_output, _ = self.feat_attention(Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2))
        attn_output = attn_output.transpose(1, 2)
        multihead_message = self.multihead_message(attn_output)

        final_message = torch.concat([Q, multihead_message], dim=1)
        final_message = self.final_message(final_message)

        # opposite_feature = None : Intra frame self attention
        # opposite_feature != None : Inter frame cross attention
        if opposite_feature == None:
            all_message = feature + final_message
        else:
            opposite_feature = opposite_feature.transpose(1, 2)
            all_message = opposite_feature + final_message
        # [batch, num_pts, in_dims]
        return all_message.transpose(1, 2)


class PPFLocalAndGlobal(nn.Module):
    def __init__(self):
        super(PPFLocalAndGlobal, self).__init__()
        self.ppfnetLocalGlobal = PPFNetLocalGlobal(emb_dims=96, radius=0.3,
                                                   num_neighbors=64, farthest_subsample_numpts=512)
        self.attentionblock = AttentionBlock(in_dims=96, num_heads=1, drop_out=0.0)

        self.num_sk_iter = 5
        self.add_slack = False
        self._EPS = 1e-5  # To prevent division by zero
        self.num_attention_layers = 2

        bin_score = nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def compute_affinity(self, feat_distance, alpha=0.5):
        """Compute logarithm of Initial match matrix values, i.e. log(m_jk)"""
        if isinstance(alpha, float):
            hybrid_affinity = -1 * (feat_distance - alpha)
        else:
            hybrid_affinity = -1 * (feat_distance - alpha[:, None, None])
        return hybrid_affinity

    def forward(self, ps, pt, ns, nt):
        # [batch, num_pts, in_dims = 96]
        ps_feat, ps_far_sample, ns_far_sample = self.ppfnetLocalGlobal(ps, ns)
        pt_feat, pt_far_sample, nt_far_sample = self.ppfnetLocalGlobal(pt, nt)

        # attention blocks
        for i in range(self.num_attention_layers):
            ps_feat = self.attentionblock(ps_feat)
            pt_feat = self.attentionblock(pt_feat)
            ps_feat = self.attentionblock(ps_feat, pt_feat)
            pt_feat = self.attentionblock(pt_feat, ps_feat)

        # RPMNet operation
        # [batch, num_pts, num_pts]
        feat_distance = match_features(ps_feat, pt_feat)
        affinity = self.compute_affinity(feat_distance)
        log_perm_matrix = sinkhorn(affinity, n_iters=self.num_sk_iter, slack=self.add_slack)
        perm_matrix = torch.exp(log_perm_matrix)
        weighted_pt = perm_matrix @ pt_far_sample / (torch.sum(perm_matrix, dim=2, keepdim=True) + self._EPS)

        # compute transform and transform points
        transform = compute_rigid_transform(ps_far_sample, weighted_pt,
                                            weights=torch.clamp(torch.sum(perm_matrix, dim=2),0,1))

        return transform


if __name__ == '__main__':
    data = np.loadtxt('../../test/data/airplane_0627.txt', delimiter=',')
    data = data[:512, 0:6]
    pose = utils.data_utils.random_pose(45, 0.5)
    pt = data[:, 0:3]
    nt = data[:, 3:6]
    ps = se3.transform_np(pose, pt)
    ns = se3.transform_np(pose, nt)


    def to_torch(data):
        return torch.from_numpy(data.astype('float32')).unsqueeze(0)


    ppflocalandglobal = PPFLocalAndGlobal()
    G = ppflocalandglobal(torch.rand((2, 512, 3)), torch.rand((2, 512, 3)), torch.rand((2, 512, 3)),
                          torch.rand((2, 512, 3)))

    R, t = se3.decompose_trans(G)
    print(torch.squeeze(t))
    print(t)

    # data = torch.rand((2, 512, 96))
    # intraframe = AttentionBlock(in_dims=96)
    # print(intraframe(data).shape)
