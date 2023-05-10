import torch
import torch.nn as nn
from RIGARework.ppf_local_global import PPFLocalGlobalNet


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
        self.ppfnet = PPFLocalGlobalNet(emb_dims=96, radius=0.3, num_neighbors=64, use_global=True)

    def forward(self, ps, pt, ns, nt):
        # [batch, num_pts, in_dims = 96]
        ps_feat = self.ppfnet(ps, ns)
        pt_feat = self.ppfnet(pt, nt)

        return ps_feat, pt_feat


if __name__ == '__main__':
    ps = torch.rand((2, 1024, 3))
    pt = torch.rand((2, 1024, 3))
    ns = torch.rand((2, 1024, 3))
    nt = torch.rand((2, 1024, 3))
    model = PPFLocalAndGlobal()
    model(ps, pt, ns, nt)
