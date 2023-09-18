import numpy as np
import torch
import se_math.se3
from utils.data_utils import farthest_avg_subsample_points, random_pose
import open3d as o3d

def load_data(fname_txt):
    # obj
    template_data = np.fromfile(fname_txt, dtype=np.float32).reshape(-1, 4)
    points = template_data[:, :3]
    return points

data_o = load_data('006062.bin')
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