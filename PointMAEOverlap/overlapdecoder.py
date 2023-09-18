import torch.nn as nn
import torch
import numpy as np


class OverlapDecoder(nn.Module):
    def __init__(self, embed_dim=384):
        super(OverlapDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.masknet = nn.Sequential(nn.Conv1d(self.embed_dim * 2, 1024, 1), nn.ReLU(),
                                     nn.Conv1d(1024, 512, 1), nn.ReLU(),
                                     nn.Conv1d(512, 256, 1), nn.ReLU(),
                                     nn.Conv1d(256, 128, 1), nn.ReLU(),
                                     nn.Conv1d(128, 1, 1), nn.Sigmoid())

    def forward(self, pt1feat, pt2feat):
        B, N, C = pt1feat.shape
        pt1feat_pool = torch.max(pt1feat, 1)[0].unsqueeze(1)
        feat = torch.cat([pt2feat, pt1feat_pool.repeat(1, N, 1)], dim=2)
        mask = self.masknet(feat.transpose(1, 2)).transpose(1, 2)
        mask = mask.view(B, -1)
        return torch.where(mask > 0.5, 1, 0)


if __name__ == '__main__':
    pt1feat = torch.rand((2, 716, 384))
    pt2feat = torch.rand((2, 716, 384))
    overlap = OverlapDecoder()
    print(overlap(pt1feat, pt2feat).shape)
    print(overlap(pt1feat,pt2feat))

    p1 = np.load('./overlap_data_test/p1.npy')
    p2 = np.load('./overlap_data_test/p2.npy')
    p2 += 0.02
    p1mask = np.load('./overlap_data_test/p1mask.npy')
    p2mask = np.load('./overlap_data_test/p2mask.npy')
    p1 = torch.from_numpy(p1).unsqueeze(0)
    p1 = torch.cat([p1, p1 + 0.03])
    p2 = torch.from_numpy(p2).unsqueeze(0)
    p2 = torch.cat([p2, p2])
    p1mask = torch.from_numpy(p1mask).unsqueeze(0)
    p1mask = torch.cat([p1mask, p1mask])
    p2mask = torch.from_numpy(p2mask).unsqueeze(0)
    p2mask = torch.cat([p2mask, p2mask])

    from metrics.benchmarks import evaluate_mask
    print(evaluate_mask(overlap(pt1feat,pt2feat),p1mask))


