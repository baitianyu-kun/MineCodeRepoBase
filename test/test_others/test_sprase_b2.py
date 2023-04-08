import numpy as np
import torch
import utils.data_utils as du
from models.DGCNN.Transformerdgcnn.dgcnn_transformer import knn

data = np.loadtxt('../data/airplane_0010.txt', delimiter=',')[:, 0:3]
data = du.farthest_avg_subsample_points(data, 5)
data = torch.from_numpy(data).unsqueeze(0).transpose(1, 2)

data2 = np.loadtxt('../data/airplane_0627.txt', delimiter=',')[:, 0:3]
data2 = du.farthest_avg_subsample_points(data2, 5)
data2 = torch.from_numpy(data2).unsqueeze(0).transpose(1, 2)

data_batch2 = torch.concat([data, data2])

knn_k = 4
# idx (batch, num_pts, knn_k)
idx = knn(data_batch2, knn_k)
edge_list = idx
batches, num_nodes, _ = edge_list.shape
adj = np.zeros((batches, num_nodes, num_nodes))
print(edge_list)
for batch in range(batches):
    for i in range(num_nodes):
        adj[batch][i][edge_list[batch][i]] = 1
print(adj)