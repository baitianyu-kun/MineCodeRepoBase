import numpy as np
import torch
from se_math.se3 import *
from utils.data_utils import *

data1 = np.loadtxt('../data/airplane_0627.txt', delimiter=',')[:10, 0:3]
data2 = np.loadtxt('../data/airplane_0010.txt', delimiter=',')[:10, 0:3]
data_batch2=torch.concat([torch.from_numpy(data1).unsqueeze(0),torch.from_numpy(data2).unsqueeze(0)])
pose_batch2=torch.concat([torch.from_numpy(random_pose(60,0.1)).unsqueeze(0),
                          torch.from_numpy(random_pose(60,0.1)).unsqueeze(0)])
data_trans=transform_torch(pose_batch2,data_batch2)
print(data_trans)
