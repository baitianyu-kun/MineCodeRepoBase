import torch
import torch.nn as nn

'''
from ''Learning to Find Good Correspondences''
and it was made to point cloud using model
'''


# actually this is equals to torch.nn.InstanceNorm1d
class ContextNormalization(nn.Module):
    def __init__(self):
        super(ContextNormalization, self).__init__()

    def forward(self, x):
        var_eps = 1e-3
        mean = torch.mean(x, 2, keepdim=True)
        variance = torch.var(x, 2, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + var_eps)
        return x


class PointCN(nn.Module):
    def __init__(self, in_dim=3, num_layers=2, num_channels=128, act_pos='post'):
        super(PointCN, self).__init__()
        assert act_pos == 'pre' or act_pos == 'post'

        modules = [nn.Conv1d(in_dim, num_channels, kernel_size=1, bias=True)]
        for i in range(num_layers):
            if act_pos == 'pre':
                modules.append(ContextNormalization())
                modules.append(nn.BatchNorm1d(num_channels))
                modules.append(nn.ReLU(inplace=True))
                modules.append(nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True))
            else:
                modules.append(nn.Conv1d(num_channels, num_channels, kernel_size=1, bias=True))
                modules.append(ContextNormalization())
                modules.append(nn.BatchNorm1d(num_channels))
                modules.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        features = self.encoder(x)
        return features


if __name__ == '__main__':
    pointcn = PointCN()
    data = torch.rand((2, 3, 1024))
    print(pointcn(data).shape)
