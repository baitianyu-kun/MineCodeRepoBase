import numpy as np
import torch
import utils.data_utils as du
import utils.visual as vi
from models.dgcnn_transformer import knn
import scipy

data = np.loadtxt('../data/airplane_0010.txt', delimiter=',')[:, 0:3]
data = du.farthest_avg_subsample_points(data, 50)
data = torch.from_numpy(data).unsqueeze(0).transpose(1, 2)
knn_k = 4
idx = knn(data, knn_k)
edgelist = idx[0]
num_nodes = 50
adj = np.zeros((num_nodes, num_nodes))
print(edgelist)
# 第0号中[0, 47, 37, 29]，说明第0号和0, 47, 37, 29相连接(0和0，即包括他自己)
for i in range(len(edgelist)):
    for j in edgelist[i]:
        adj[i][j] = 1
