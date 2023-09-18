import numpy as np
import torch
import se_math.se3
from utils.data_utils import farthest_avg_subsample_points, random_pose
import open3d as o3d


# kitti_obj:000000.bin
# kitti_odo:000001.bin
def load_data(fname_txt):
    # obj
    template_data = np.fromfile(fname_txt, dtype=np.float32).reshape(-1, 4)
    points = template_data[:, :3]
    return points


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
    cam_control.set_front(front=(0.28251738518096992, 0.82247859157527492, 0.49367286078015243))
    cam_control.set_lookat((0.16524678468704224, 0.30312192440032959, 0.14933159947395325))
    cam_control.set_up((-0.1016000559288687, -0.48608735586055885, 0.86798416524020838))
    cam_control.set_zoom(0.41999999999999971)
    # cam_control.field_of_view(60)
    vis.run()
    vis.destroy_window()


files = ['New Folder/002583.bin','New Folder/002584.bin','New Folder/002585.bin','New Folder/002586.bin'
         ,'New Folder/002587.bin','New Folder/002588.bin','New Folder/002589.bin']
for file in files:
    data_o = load_data(file)

    transforms_gt = np.load('transforms_igt_matrix.npy')

    data = farthest_avg_subsample_points(data_o, npoint=4096)
    data_gt = se_math.se3.transform_np(transforms_gt, data)
    pose = random_pose(0, 0.9)
    data_trans = se_math.se3.transform_np(pose, data)
    show(data, data_gt, data_trans, light_mode=True)

