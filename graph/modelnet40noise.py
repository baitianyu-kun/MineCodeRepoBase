from random import sample

import numpy as np
import open3d as o3d
import torch

data1 = np.loadtxt('airplane_0627.txt', delimiter=',')[:, 0:3]
# data_noise = torch.normal(torch.from_numpy(data1), 0.02).numpy()
pts_num=data1.shape[0]
sample_num=int(pts_num*0.3)
num = sample(range(1, pts_num), sample_num)
data_density=data1[num,:]

pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(data1)

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(data_density)

# 创建窗口对象
vis = o3d.visualization.Visualizer()
# 设置窗口标题
vis.create_window(window_name="Test", width=1920, height=1080)
# 设置点云大小
vis.get_render_option().point_size = 5
cam_control = vis.get_view_control()



vis.add_geometry(pcd1)

cam_control.set_front(front=(-0.47447046287774686, 0.75630054827113813, 0.45043008396570572))
cam_control.set_lookat((0.0049955112722364636, -0.043853213903476324, -0.00086175537627264376))
cam_control.set_up((0.20029393013636757, 0.59102774019126891, -0.78138886085925874))
cam_control.set_zoom(0.69999999999999996)

# cam_control.set_front(front=(-0.95058292287819579, -0.013286794611390939, -0.31018634370536219))
# cam_control.set_lookat((-0.00014999999999998348, -0.091399999999999981, -0.00030000000000002247))
# cam_control.set_up((-0.11367178639353176, 0.94460663569162251, 0.30789126130089783))
# cam_control.set_zoom(1.0600000000000003)

# cam_control.set_front(front=(-0.30408715574622552, 0.37717461295610616, -0.87479729826490793))
# cam_control.set_lookat((-0.00014999999999998348, -0.091399999999999981, -0.00030000000000002247))
# cam_control.set_up((0.064152244629373467, 0.92430639805796511,0.37622090853396695))
# cam_control.set_zoom(1.0600000000000003)

vis.run()
vis.destroy_window()
