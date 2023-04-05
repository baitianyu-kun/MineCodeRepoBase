import numpy as np
import torch
import utils.data_utils as du
from models.Transformerdgcnn.dgcnn_transformer import knn

data = np.loadtxt('../data/airplane_0010.txt', delimiter=',')[:, 0:3]
data = du.farthest_avg_subsample_points(data, 5)
data = torch.from_numpy(data).unsqueeze(0).transpose(1, 2)

data2 = np.loadtxt('../data/airplane_0627.txt', delimiter=',')[:, 0:3]
data2 = du.farthest_avg_subsample_points(data2, 5)
data2 = torch.from_numpy(data2).unsqueeze(0).transpose(1, 2)

data_batch2=torch.concat([data,data2])

knn_k = 4
idx = knn(data, knn_k)
edgelist = idx[0]
num_nodes = len(edgelist)
adj = np.zeros((num_nodes, num_nodes))
print(edgelist)
# 第0号中[0, 47, 37, 29]，说明第0号和0, 47, 37, 29相连接(0和0，即包括他自己)
for i in range(num_nodes):
    for j in edgelist[i]:
        adj[i][j] = 1

# edgelist([[0, 4, 3, 2],
#         [1, 2, 3, 0],
#         [2, 1, 0, 4],
#         [3, 0, 1, 2],
#         [4, 0, 2, 3]]) -> adj2[i][0,4,3,2]=1 is ok too
adj2 = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    adj2[i][edgelist[i]] = 1

