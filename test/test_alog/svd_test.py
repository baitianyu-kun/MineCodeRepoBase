import torch
import numpy as np
from utils.data_utils import random_pose
import se_math.se3 as se3
from utils.visual import *
from algo import svd

# data1 = np.loadtxt('../data/airplane_0010.txt', delimiter=',')[0:1024, 0:3]
# pose1 = random_pose(40, 0.02)
# data2 = np.loadtxt('../data/airplane_0627.txt', delimiter=',')[0:1024, 0:3]
# pose2 = random_pose(40, 0.02)
# data_all = torch.concat([torch.from_numpy(data1).unsqueeze(0), torch.from_numpy(data2).unsqueeze(0)])
# pose_all = torch.concat([torch.from_numpy(pose1).unsqueeze(0), torch.from_numpy(pose2).unsqueeze(0)])
# data_all_trans=se3.transform_torch(pose_all,data_all)
# print(svd.compute_rigid_transform(data_all,data_all_trans))
# print(pose_all)

