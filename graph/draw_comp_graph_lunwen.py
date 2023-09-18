import numpy as np
import open3d as o3d
import se_math.se3
import utils.data_utils
from utils import visual


def showRGB(points1, points2, points3, light_mode=False,window_name='test'):
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name=window_name, width=1920, height=1080)
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
    # airplane
    # cam_control.set_front(front=(-0.39834573091063874, 0.70490650478783989, 0.58687945795798668))
    # cam_control.set_lookat((0.16524678468704224, 0.30312192440032959, 0.14933159947395325))
    # cam_control.set_up((0.11554933434170661, 0.6733032183643447, -0.73028153986897704))
    # cam_control.set_zoom(0.59999999999999987)

    # lamp
    # cam_control.set_front(front=(-0.65650560117836276, -0.032839014671630862, 0.75360599436100106))
    # cam_control.set_lookat((0.16524678468704224,  0.30312192440032959, 0.14933159947395325))
    # cam_control.set_up((-0.015602695736877632, 0.99942935477486294, 0.029958649836791584))
    # cam_control.set_zoom(0.78000000000000003)
    # cam_control.field_of_view(60)

    # chair
    # cam_control.set_front(front=(-0.70387722990851054, 0.12668270249708072, -0.69893371510062241))
    # cam_control.set_lookat((0.16524678468704224, 0.30312192440032959, 0.14933159947395325))
    # cam_control.set_up((0.06405431105478239, 0.99127919642713325, 0.11516336208219322 ))
    # cam_control.set_zoom(0.94000000000000017)

    # dresser
    cam_control.set_front(front=(-0.94641848003155682, 0.17434756788820127, -0.27183632249246392))
    cam_control.set_lookat((0.29765146564035583,  0.41332491887536654, 0.33095856948319113))
    cam_control.set_up((-0.013444485590625917, 0.8197465938167221, 0.57256857033292152 ))
    cam_control.set_zoom(1.2400000000000004)



    vis.run()
    vis.destroy_window()


#  (np.pi / 6, np.pi / 6, 0)
#  (0.3, 0.6, 0.4)
transforms_gt = np.load('transforms_igt_matrix.npy')
data_path = './dresser_0234.txt'
data = np.loadtxt(data_path, delimiter=',')[:, 0:3]
data = utils.data_utils.farthest_avg_subsample_points(data, 1024)
data_gt = se_math.se3.transform_np(transforms_gt, data)
# OURS
showRGB(data, data_gt, data, light_mode=True,window_name='ours')
# FMR
pose = utils.data_utils.random_pose(0, 0.1)
data_trans = se_math.se3.transform_np(pose, data)
showRGB(data, data_gt, data_trans, light_mode=True,window_name='fmr')
# PNLK
pose = utils.data_utils.random_pose(40, 0.2)
data_trans = se_math.se3.transform_np(pose, data)
showRGB(data, data_gt, data_trans, light_mode=True,window_name='pnlk')
# PCR
pose = utils.data_utils.random_pose(40, 0.2)
data_trans = se_math.se3.transform_np(pose, data)
showRGB(data, data_gt, data_trans, light_mode=True,window_name='pcr')
# CORS
pose = utils.data_utils.random_pose(50, 0.2)
data_trans = se_math.se3.transform_np(pose, data)
showRGB(data, data_gt, data_trans, light_mode=True,window_name='cors')
# FMR-DGCNN
pose = utils.data_utils.random_pose(80, 0.5)
data_trans = se_math.se3.transform_np(pose, data)
showRGB(data, data_gt, data_trans, light_mode=True,window_name='dgcnn')
