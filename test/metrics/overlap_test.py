import copy
import numpy as np
from utils.data_utils import to_o3d_pcd,farthest_neighbour_subsample_points2
from utils.visual import *
from metrics import overlap


data1 = np.loadtxt('../data/airplane_0010.txt', delimiter=',')[:, 0:3]
data2,_=farthest_neighbour_subsample_points2(data1,int(0.45*data1.shape[0]))
data1 = to_o3d_pcd(data1)
data2 = to_o3d_pcd(data2)
print(overlap.get_overlap_ratio(data1, data2))
