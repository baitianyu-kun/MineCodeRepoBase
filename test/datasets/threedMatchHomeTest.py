import glob
import os.path
import numpy as np
import open3d as o3d
import utils.visual as vi


def normalize_pc(point_cloud):
    centroid = np.mean(point_cloud, axis=0)
    point_cloud -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(point_cloud) ** 2, axis=-1)))
    point_cloud /= furthest_distance
    return point_cloud


data_dir = 'D:\\dataset\\sun3d-home_at-home_at_scan1_2013_jan_1'
files = glob.glob(os.path.join(data_dir, 'cloud_bin_*.ply'))
num_points = 4096
all_data = []
for file in files:
    pc = o3d.io.read_point_cloud(file)
    points = normalize_pc(np.array(pc.points))
    points_idx = np.arange(points.shape[0])
    np.random.shuffle(points_idx)
    points = points[points_idx[:num_points], :]
    all_data.append(points)

vi.show(all_data[0],light_mode=True)