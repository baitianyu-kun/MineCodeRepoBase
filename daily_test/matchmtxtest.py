import torch
import numpy as np
import torch.nn as nn

x = torch.from_numpy(np.load('x.npy')).unsqueeze(0)
y = torch.from_numpy(np.load('y.npy')).unsqueeze(0)


class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.conv1 = nn.Conv1d(3, 512, 1)

    def forward(self, x):
        x = self.conv1(x)
        return x


net = mlp()
x_feat = net(x.transpose(1, 2)).transpose(1, 2)
y_feat = net(y.transpose(1, 2)).transpose(1, 2)
match_mtx = torch.bmm(x_feat, y_feat.transpose(1, 2))
print(match_mtx.shape)
print(x_feat.shape)
print(y_feat.shape)

y_corrs = torch.bmm(match_mtx, y)
print(y_corrs.shape)
print(x.shape)
from algo.svd import compute_rigid_transform

G = compute_rigid_transform(x, y_corrs)
print(G)
