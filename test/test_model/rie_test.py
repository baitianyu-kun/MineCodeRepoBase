import numpy as np
from scipy.spatial.distance import minkowski
from sklearn.neighbors import NearestNeighbors

import se_math.se3
import utils.data_utils
from utils import visual
from utils import datasets


def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=1024):
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    random_p2 = random_p1
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :], pointcloud2[idx2, :]


# data = np.random.rand(1024, 3)
data, label = datasets.load_data('D:\\dataset\\sun3d-home_at-home_at_scan1_2013_jan_1', partition='test',
                                 file_type='3DMatchHome')
data = data[0]
pose = utils.data_utils.random_pose(max_angle=40, max_trans=0.5)
data2 = se_math.se3.transform_np(pose, data)
data, data2 = farthest_subsample_points(data, data2, num_subsampled_points=8000)
visual.show(data2,data)
data2 = se_math.se3.transform_np(se_math.se3.inverse_np(pose), data2)
data2+=0.001
visual.show(data2, data)
