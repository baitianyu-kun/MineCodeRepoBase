import torch
import numpy as np
from utils import data_utils
from se_math import se3, so3
from utils import visual

# points1 = np.loadtxt('../data/airplane_0010.txt', delimiter=',')[:, 0:3]
# pose = data_utils.random_pose(100, 0.1)
# pose = torch.from_numpy(pose)
# points1_transformed = se3.transform(pose.unsqueeze(0), torch.from_numpy(points1)).numpy()
#
# source, source_mask = data_utils.Farthest_Point_Sampling(points1,
#                                                          int(0.2 * points1.shape[0]))
#
# p1, p2 = data_utils.farthest_neighbour_subsample_points(points1, points1_transformed, int(0.2 * points1.shape[0]))
#
# visual.show(p1,p2)

# sampled_pts=data_utils.random_select_points(points1,10)
# visual.show(sampled_pts)

vec1=np.random.random((3,3))
vec2=np.random.random((3,3))
print(data_utils.angle(vec1,vec2))
print(data_utils.angle_torch(torch.from_numpy(vec1),torch.from_numpy(vec2)))
