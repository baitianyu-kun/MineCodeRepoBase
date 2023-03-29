import torch
import utils.data_utils as du
from se_math import so3
import numpy as np
from utils.visual import *
from copy import deepcopy

data = np.loadtxt('../data/airplane_0627.txt', delimiter=',')[:, 0:3]
data2 = np.loadtxt('../data/airplane_0010.txt', delimiter=',')[:, 0:3]
rotate_mtx = du.random_rotation(60)
rotate_mtx_torch = torch.from_numpy(rotate_mtx)
data_torch = torch.from_numpy(data)
data2_torch = torch.from_numpy(data2)
data_2_batch = torch.concat([data_torch.unsqueeze(0), data2_torch.unsqueeze(0)])
rotate_mtx_2_batch=torch.concat([rotate_mtx_torch.unsqueeze(0),rotate_mtx_torch.unsqueeze(0)])
transform_origin=so3.transform(rotate_mtx_2_batch,data_2_batch.transpose(1,2)).transpose(1,2)
transform_torch=so3.transform_torch(rotate_mtx_2_batch,data_2_batch)
print(transform_origin)
print(transform_torch)
