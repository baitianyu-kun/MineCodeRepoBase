import random

import open3d as o3d
import numpy as np


def show(*inputs, light_mode=False):
    palette = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1]]
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name="Test", width=1920, height=1080)
    # 设置点云大小
    vis.get_render_option().point_size = 5
    # 设置颜色背景为黑色
    opt = vis.get_render_option()
    if light_mode:
        opt.background_color = np.asarray([1, 1, 1])
    else:
        opt.background_color = np.asarray([0, 0, 0])
    # 设置相机
    cam_control = vis.get_view_control()
    color_index = 0
    for input in inputs:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(input)
        pcd.paint_uniform_color(palette[color_index])
        color_index += 1
        vis.add_geometry(pcd)
    # cam_control.set_front(front=(1, 0, 0))
    vis.run()
    vis.destroy_window()


def showlist(lists, light_mode=False):
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name="Test", width=1920, height=1080)
    # 设置点云大小
    vis.get_render_option().point_size = 5
    # 设置颜色背景为黑色
    opt = vis.get_render_option()
    if light_mode:
        opt.background_color = np.asarray([1, 1, 1])
    else:
        opt.background_color = np.asarray([0, 0, 0])
    # 设置相机
    cam_control = vis.get_view_control()
    for input in lists:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(input)
        pcd.paint_uniform_color([random.uniform(0, 1), 0, random.uniform(0, 1)])
        vis.add_geometry(pcd)
    # cam_control.set_front(front=(1, 0, 0))
    vis.run()
    vis.destroy_window()


def showRGB(points1, points2, points3, light_mode=False):
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name="Test", width=1920, height=1080)
    # 设置点云大小
    vis.get_render_option().point_size = 5

    # 设置颜色背景为黑色
    opt = vis.get_render_option()
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
    # cam_control.set_front(front=(1, 0, 0))
    vis.run()
    vis.destroy_window()


def showRGBY(points1, points2, points3, points4, light_mode=False):
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name="Test", width=1920, height=1080)
    # 设置点云大小
    vis.get_render_option().point_size = 5

    # 设置颜色背景为黑色
    opt = vis.get_render_option()
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

    pcd4 = o3d.geometry.PointCloud()
    pcd4.points = o3d.utility.Vector3dVector(points4)
    pcd4.paint_uniform_color([0, 1, 1])
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    vis.add_geometry(pcd3)
    vis.add_geometry(pcd4)
    # 设置相机视角朝前
    # cam_control.set_front(front=(1, 0, 0))
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    # data = np.genfromtxt('../data/1a04e3eab45ca15dd86060f189eb133.txt', delimiter=' ')[:, 0:3]
    # data2 = np.genfromtxt('../data/1a680e3308f2aac544b2fa2cac0778f5.txt', delimiter=' ')[:, 0:3]
    # show(data, data2)

    showRGB(np.load('../result/best_model_predict/epoch_6/showed_p0.npy'),
            np.load('../result/best_model_predict/epoch_6/showed_p1.npy'),
            np.load('../result/best_model_predict/epoch_6/show_p_use_est_g.npy'), )
