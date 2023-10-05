import numpy as np
import open3d as o3d

from utils.data_utils import farthest_avg_subsample_points

p = o3d.io.read_point_cloud('cloud_bin_kitchen.ply')
p = np.array(p.points)
p = farthest_avg_subsample_points(p, npoint=10000)
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(p)

# 创建窗口对象
vis = o3d.visualization.Visualizer()
# 设置窗口标题
vis.create_window(window_name="Test", width=1920, height=1080)
# 设置点云大小
vis.get_render_option().point_size = 5
vis.add_geometry(pcd1)
vis.run()
vis.destroy_window()
