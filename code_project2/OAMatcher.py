import math

import numpy as np
import torch
import torch.nn as nn
from feature_extract import DGCNN
from transformer_module import Transformer


class OAMatcherNet(nn.Module):
    def __init__(self):
        super(OAMatcherNet, self).__init__()
        self.feature_extractor = DGCNN(emb_dims=512)
        self.transformer_module = Transformer(emb_dims=512, n_blocks=1, dropout=0.0, ff_dims=1024, n_heads=4)

    def forward(self, ps, pt):
        ps_feature = self.feature_extractor(ps)
        pt_feature = self.feature_extractor(pt)
        ps_emb, pt_emb = self.transformer_module(ps_feature, pt_feature)
        dim_k = ps_emb.shape[1]
        ps_emb = ps_feature + ps_emb
        pt_emb = pt_feature + pt_emb
        match_mtx = torch.bmm(ps_emb.transpose(1, 2), pt_emb) / math.sqrt(dim_k)
        # row softmax
        match_mtx_ps = torch.softmax(match_mtx, dim=2)
        match_mtx_pt = torch.softmax(match_mtx, dim=1)

        import cv2 as cv
        cv.imshow('test',match_mtx_ps[0].detach().cpu().numpy())
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    OAMatcher = OAMatcherNet().cuda()
    data1 = torch.rand((2, 3, 1024)).cuda()
    data2 = torch.rand((2, 3, 1024)).cuda()
    OAMatcher(data1, data2)
