import torch
import torch.nn as nn


# decoder network
class Decoder(torch.nn.Module):
    def __init__(self, num_points=2048, bottleneck_size=1024):
        super(Decoder, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size // 4)
        self.fc1 = torch.nn.Linear(self.bottleneck_size, bottleneck_size)
        self.fc2 = torch.nn.Linear(self.bottleneck_size, bottleneck_size // 2)
        self.fc3 = torch.nn.Linear(bottleneck_size // 2, bottleneck_size // 4)
        self.fc4 = torch.nn.Linear(bottleneck_size // 4, self.num_points * 3)
        self.th = torch.nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = torch.nn.functional.relu(self.bn1(self.fc1(x)))
        x = torch.nn.functional.relu(self.bn2(self.fc2(x)))
        x = torch.nn.functional.relu(self.bn3(self.fc3(x)))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, self.num_points).transpose(1, 2).contiguous()
        return x


# a global function to generate mlp layers
def _mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers


# a global function to flatten a feature
def flatten(x):
    return x.view(x.size(0), -1)


# a global function to calculate max-pooling
def symfn_max(x):
    # [B, K, N] -> [B, K, 1]
    a = torch.nn.functional.max_pool1d(x, x.size(-1))
    return a


# a class to generate MLP network
class MLPNet(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """

    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        list_layers = _mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        out = self.layers(inp)
        return out


class PointNet(torch.nn.Module):
    def __init__(self, dim_k=1024):
        super().__init__()
        scale = 1
        mlp_h1 = [int(64 / scale), int(64 / scale)]
        mlp_h2 = [int(64 / scale), int(128 / scale), int(dim_k / scale)]

        self.h1 = MLPNet(3, mlp_h1, b_shared=True).layers
        self.h2 = MLPNet(mlp_h1[-1], mlp_h2, b_shared=True).layers
        self.sy = symfn_max

    def forward(self, points):
        """ points -> features
            [B, N, 3] -> [B, K]
        """
        # for pointnet feature extraction
        x = points.transpose(1, 2)  # [B, 3, N]
        x = self.h1(x)
        x = self.h2(x)  # [B, K, N]
        # x = flatten(self.sy(x))
        return x, flatten(self.sy(x))
