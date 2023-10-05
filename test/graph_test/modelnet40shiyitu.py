import numpy as np
import open3d as o3d

data=np.loadtxt('car_0198.txt',delimiter=',')[:,0:3]
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