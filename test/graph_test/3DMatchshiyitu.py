import glob
import os
from utils.data_utils import farthest_avg_subsample_points
import open3d as o3d
import numpy as np


def normalize_pc(point_cloud):
    centroid = np.mean(point_cloud, axis=0)
    point_cloud -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(point_cloud) ** 2, axis=-1)))
    point_cloud /= furthest_distance
    return point_cloud


def load_data(DATA_DIR):
    num_points = 10000
    all_data = []
    all_label = []
    for h5_name in glob.glob(
            os.path.join(DATA_DIR, 'cloud_bin_*.ply')):
        pc = o3d.io.read_point_cloud(h5_name)
        points = normalize_pc(np.array(pc.points))
        # 采样10000个点
        points_idx = np.arange(points.shape[0])
        np.random.shuffle(points_idx)
        points = points[points_idx[:num_points], :]
        all_data.append(points)
    return np.array(all_data), np.array(all_label)


datas, labels = load_data('D:\\dataset\\sun3d-home_at-home_at_scan1_2013_jan_1')
data_o = datas[5]
data = farthest_avg_subsample_points(data_o, npoint=10000)
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(data)

# 创建窗口对象
vis = o3d.visualization.Visualizer()
# 设置窗口标题
vis.create_window(window_name="Test", width=1920, height=1080)
# 设置点云大小
vis.get_render_option().point_size = 5
vis.add_geometry(pcd1)
vis.run()
vis.destroy_window()
