import numpy as np
import torch
import os
import open3d as o3d
import glob

import se_math.se3
from utils.data_utils import farthest_avg_subsample_points,random_pose



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


def show(points1, points2, points3, light_mode=False):
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name="Test", width=1920, height=1080)
    # 设置点云大小
    vis.get_render_option().point_size = 5

    # 设置颜色背景为黑色
    opt = vis.get_render_option()
    # opt.point_rendering_method = o3d.visualization.PointRenderingOption.PointRenderingMethod.Circles
    if light_mode:
        opt.background_color = np.asarray([1, 1, 1])
    else:
        opt.background_color = np.asarray([0, 0, 0])
    # opt.point_size=10.0
    # 设置相机
    cam_control = vis.get_view_control()
    # 源
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd1.paint_uniform_color([1, 0, 0])
    # 模板
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd2.paint_uniform_color([0, 1, 0])
    # 手动变换，应该让后两个接近
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(points3)
    pcd3.paint_uniform_color([0, 0, 1])
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    vis.add_geometry(pcd3)
    # 设置相机视角朝前
    cam_control.set_front(front=(-0.51608616977823951, -0.25794263708056947, 0.81677454743616229))
    cam_control.set_lookat((0.16524678468704224, 0.30312192440032959, 0.14933159947395325))
    cam_control.set_up((-0.065092209173675711, 0.96263137969286194, 0.2628760756240619))
    cam_control.set_zoom(0.84000000000000008)
    # cam_control.field_of_view(60)
    vis.run()
    vis.destroy_window()


datas, labels = load_data('D:\\dataset\\sun3d-home_at-home_at_scan1_2013_jan_1')
transforms_gt = np.load('transforms_igt_matrix.npy')
for i in range(20,30):
    data_o = datas[i]
    data = farthest_avg_subsample_points(data_o, npoint=4096)
    data_gt=se_math.se3.transform_np(transforms_gt,data)
    pose = random_pose(0, 0.06)
    data_trans = se_math.se3.transform_np(pose, data)
    show(data, data_gt, data_trans, light_mode=True)



