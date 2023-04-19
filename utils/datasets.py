import h5py
import numpy as np
import torch
import glob
import open3d as o3d
import os


def normalize_pc(point_cloud):
    centroid = np.mean(point_cloud, axis=0)
    point_cloud -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(point_cloud) ** 2, axis=-1)))
    point_cloud /= furthest_distance
    return point_cloud


def load_data(DATA_DIR, partition, file_type='modelnet40'):
    # 读取训练集or测试集
    if file_type == '3DMatch7':
        num_points = 4096
        file_name = '3DMatch/7-scenes-redkitchen/' + partition
        all_data = []
        all_label = []
        for h5_name in glob.glob(os.path.join(DATA_DIR, file_name, 'cloud_bin_*.ply')):
            pc = o3d.io.read_point_cloud(h5_name)
            points = normalize_pc(np.array(pc.points))
            # 采样10000个点
            points_idx = np.arange(points.shape[0])
            np.random.shuffle(points_idx)
            points = points[points_idx[:num_points], :]
            all_data.append(points)
        return np.array(all_data), np.array(all_label)
    elif file_type == '3DMatch_all':
        num_points = 4096
        file_name = '3dmatch_all'
        all_data = []
        all_label = []
        DATA_FILES = {
            'train': os.path.join(DATA_DIR, file_name, 'split/train_3dmatch.txt'),  # 1629
            'test': os.path.join(DATA_DIR, file_name, 'split/test_3dmatch.txt'),  # 426
        }
        with open(DATA_FILES[partition], 'r') as f:
            datafile = f.read().split()
        for npz_name in datafile:
            for npz in glob.glob(os.path.join(DATA_DIR, file_name, '{}*.npz'.format(npz_name))):
                f = np.load(npz)
                cur_points = f['pcd'][:].astype(np.float32)
                # cur_colors = f['color'][:].astype(np.float32)
                data = np.concatenate([cur_points], axis=-1).astype(np.float32)
                if data.shape[0] >= num_points:
                    all_data.append(data[:num_points, :])
        return np.array(all_data), np.array(all_label)
    elif file_type == 'Kitti_obj':
        num_points = 4096
        file_name = 'kitti_object/velodyne'
        DATA_FILES = {
            'train': os.path.join(DATA_DIR, file_name + '/training/velodyne'),
            'test': os.path.join(DATA_DIR, file_name + '/testing/velodyne'),
        }
        all_data = []
        all_label = []
        for fname_txt in glob.glob(os.path.join(DATA_FILES[partition], "*.bin")):
            template_data = np.fromfile(fname_txt, dtype=np.float32).reshape(-1, 4)
            points = template_data[:num_points, :3]
            # points_idx = np.arange(points.shape[0])
            # np.random.shuffle(points_idx)
            # points = points[points_idx[:2048], :]
            all_data.append(points)
        return np.array(all_data), np.array(all_label)
    elif file_type == 'Kitti_odo':
        num_points = 4096
        file_name = 'KITTI_odometry/sequences/'
        DATA_FILES = {
            'train': ['00', '01', '02', '03', '04', '05'],
            'test': ['08', '09', '10'],
        }
        all_data = []
        all_label = []
        for idx in DATA_FILES[partition]:
            for fname in glob.glob(os.path.join(DATA_DIR, file_name, idx, "velodyne", "*.bin")):
                template_data = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
                # print(template_data.shape)
                points = template_data[:num_points, :3]
                # points_idx = np.arange(points.shape[0])
                # np.random.shuffle(points_idx)
                # points = points[points_idx[:2048], :]
                all_data.append(points)
                all_label.append(0)
        return np.array(all_data), np.array(all_label)
    elif file_type == 'modelnet40':
        file_name = 'modelnet40_ply_hdf5_2048'
    elif file_type == 'S3DIS':
        file_name = 'S3DIS_hdf5'
    elif file_type == 'MVP':
        if partition == 'train':
            file_name = 'mvp_partial/MVP_Train_RG.h5'
            file_name = os.path.join(DATA_DIR, file_name)
            f = h5py.File(file_name)
            data = f['src'][:].astype('float32')
        elif partition == 'test':
            file_name = 'mvp_partial/MVP_Test_RG.h5'
            file_name = os.path.join(DATA_DIR, file_name)
            f = h5py.File(file_name)
            data = f['rotated_src'][:].astype('float32')
        label = f['cat_labels'][:].astype('int64')
        f.close()
        return np.array(data), np.array(label)
    elif file_type == 'Apollo':
        num_points = 4096
        file_name = 'apollo/HighWay/' + partition + '/pcds'
        all_data = []
        all_label = []
        for h5_name in glob.glob(os.path.join(DATA_DIR, file_name, '*.pcd')):
            pc = o3d.io.read_point_cloud(h5_name)
            points = normalize_pc(np.array(pc.points))
            points_idx = np.arange(points.shape[0])
            np.random.shuffle(points_idx)
            points = points[points_idx[:num_points], :]
            all_data.append(points)
        return np.array(all_data), np.array(all_label)
    elif file_type == 'bunny':
        num_points = 4096
        file_name = 'bunny/data/'
        all_data = []
        all_label = []
        for h5_name in glob.glob(os.path.join(DATA_DIR, file_name, '*.ply')):
            pc = o3d.io.read_point_cloud(h5_name)
            points = normalize_pc(np.array(pc.points))
            points_idx = np.arange(points.shape[0])
            np.random.shuffle(points_idx)
            points = points[points_idx[:num_points], :]
            all_data.append(points)
        return np.array(all_data), np.array(all_label)
    else:
        print('Error file name!')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, file_name, 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        if file_name == 'S3DIS_hdf5':
            data = data[:, :, 0:3]
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label  # (9840, 2048, 3), (9840, 1)
