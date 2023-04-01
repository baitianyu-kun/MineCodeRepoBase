import copy

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

from utils.visual import *
from utils.data_utils import *


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


# idx is the edge_list : (num_nodes, knn_k_value) 前面是点数，后面是邻居数量，目前还需要拆开每个batch进行计算
def get_adj_matrix_batch1(edge_list):
    num_nodes = len(edge_list)
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in edge_list[i]:
            adj[i][j] = 1
    return adj


def get_graph_feature(x, k=4):
    # x = x.squeeze()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # CHANGED
    # feature = torch.cat((feature, x), dim=3)
    # feature : (batch, num_pts, neighbours, channel)
    return feature


def get_graph_feature2(x, k=4):
    # x = x.squeeze()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    # 不加batch偏移量的idx，用来先做测试，后面再加上，并把adj matrix那块也加上batch的偏移量
    origin_idx = copy.deepcopy(idx)
    batch_size, num_points, _ = idx.size()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # CHANGED
    # feature = torch.cat((feature, x), dim=3)
    # feature : (batch, num_pts, neighbours, channel)
    return feature, origin_idx


class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, act=torch.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.act = act
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        output = self.bn(output)
        return self.act(output)


class DGCNN(torch.nn.Module):
    def __init__(self, emb_dims=1024, input_shape="bnc"):
        super(DGCNN, self).__init__()
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError("Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' ")
        self.input_shape = input_shape
        self.emb_dims = emb_dims
        # 需要注意的是transformer 只能输入 seqlenth x batch x dim 形式的数据。
        self.atten1 = torch.nn.TransformerEncoderLayer(d_model=3, nhead=1, dim_feedforward=3, dropout=0.5)
        self.graphconv1 = GraphConvolution(in_features=3, out_features=3)

    def forward(self, input_data):
        if self.input_shape == "bnc":
            input_data = input_data.permute(0, 2, 1)
        if input_data.shape[1] != 3:
            raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")
        batch_size, num_dims, num_points = input_data.size()
        # torch.Size([10, 1024, 4, 3])
        # feature output : (batch, num_pts, neighbours, dims)
        output, idx = get_graph_feature2(input_data)
        # 给这几个neighbours进行平均池化，变成一个点，然后对每个点进行transformer，最后按照V2再进行一个AdjMatrix的卷积
        # avg : (batch, num_pts, dims) -> (1, 1024, 3) 经过平均聚合后的点
        output_neighbour_avg_pts = torch.mean(output, dim=2)

        # ========先在第一个batch中计算==========
        edge_list_batch1 = idx[0]
        adj_matrix_batch1 = torch.from_numpy(get_adj_matrix_batch1(edge_list_batch1)).to(torch.float32).cuda()
        # ====================

        # transformer input : (seqlength, batch, dim) -> (1024, 1, 3)
        output_neighbour_avg_pts = output_neighbour_avg_pts.permute(1, 0, 2)
        # output_transformer : (1024, 1, 3)
        output_transformer = self.atten1(output_neighbour_avg_pts.to(torch.float32))
        # Change back (1, 1024, 3)
        output_transformer = output_transformer.permute(1, 0, 2)

        # ========先在第一个batch中计算==========
        output_transformer_feature_batch1 = output_transformer[0]
        output_graph_conv = self.graphconv1(output_transformer_feature_batch1, adj_matrix_batch1)
        # ==========

        show(output_neighbour_avg_pts.permute(1, 0, 2)[0].detach().cpu().numpy(),
             output_transformer_feature_batch1.detach().cpu().numpy(),
             output_graph_conv.detach().cpu().numpy())

        return output


if __name__ == '__main__':
    # Test the code.
    # x = torch.rand((10, 1024, 3)).cuda()
    # dgcnn = DGCNN().cuda()
    # y = dgcnn(x)
    # print("\nInput Shape of DGCNN: ", x.shape, "\nOutput Shape of DGCNN: ", y.shape)

    data = np.loadtxt('../test/data/airplane_0010.txt', delimiter=',')[:, 0:3]
    data = farthest_avg_subsample_points(data, 1024)
    data = torch.from_numpy(data).unsqueeze(0).cuda()
    dgcnn = DGCNN().cuda()
    out = dgcnn(data)
