import numpy as np
import torch
from scipy.spatial.distance import minkowski
from sklearn.neighbors import NearestNeighbors
import open3d as o3d


def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd


def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.registration.Feature()
    feats.data = to_array(embedding).T
    return feats


def to_tensor(array):
    """
    Convert array to tensor
    """
    if (not isinstance(array, torch.Tensor)):
        return torch.from_numpy(array).float()
    else:
        return array


def to_array(tensor):
    """
    Conver tensor to array
    """
    if (not isinstance(tensor, np.ndarray)):
        if (tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere 生成均匀球体
    Source: https://gist.github.com/andrewbolster/10274979
    Args:
        num: Number of vectors to sample (or None if single)
    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)
    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)
    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack((x, y, z), axis=-1)


def jitter_pcd(pcd, sigma=0.01, clip=0.05):
    pcd += np.clip(sigma * np.random.randn(*pcd.shape), -1 * clip, clip)
    return pcd


def random_pose(max_angle, max_trans):
    R = random_rotation(max_angle)
    t = random_translation(max_trans)
    return np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)


def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    # Radian change to angle 弧度转换为角度
    max_angle = max_angle / 180 * np.pi
    angle = np.random.rand() * max_angle
    A = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist
    return np.expand_dims(t, 1)


def farthest_neighbour_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
    """ 随机采部分点云 不连续非完整
    Args:
        pointcloud1:
        pointcloud2:
        num_subsampled_points:
    Returns:
    """
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    random_p2 = random_p1  # np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :], pointcloud2[idx2, :]


def farthest_neighbour_subsample_points2(pointcloud1, src_subsampled_points, tgt_subsampled_points=None):
    """
    Args:
        pointcloud1:
        src_subsampled_points:
        tgt_subsampled_points:
    Returns:
    """
    # (num_points, 3)
    pointcloud1 = pointcloud1
    num_points = pointcloud1.shape[0]

    if tgt_subsampled_points is None:
        nbrs1 = NearestNeighbors(n_neighbors=src_subsampled_points, algorithm='auto',
                                 metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
        random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]])
        idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((src_subsampled_points,))
        gt_mask_src = torch.zeros(num_points).scatter_(0, torch.tensor(idx1), 1)
        return pointcloud1[idx1, :], gt_mask_src

    else:
        nbrs_src = NearestNeighbors(n_neighbors=src_subsampled_points, algorithm='auto',
                                    metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
        # 打乱点的顺序
        nbrs_tgt = NearestNeighbors(n_neighbors=tgt_subsampled_points, algorithm='auto',
                                    metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
        random = np.random.random(size=(1, 3))
        random_p1 = random + np.array([[500, 500, 500]])
        src = nbrs_src.kneighbors(random_p1, return_distance=False).reshape((src_subsampled_points,))
        mask_src = torch.zeros(num_points).scatter_(0, torch.tensor(src), 1)  # (src_subsampled_points)
        src = torch.sort(torch.tensor(src))[0]
        random_p2 = random - np.array([[500, 500, 500]])
        tgt = nbrs_tgt.kneighbors(random_p2, return_distance=False).reshape((tgt_subsampled_points,))
        mask_tgt = torch.zeros(num_points).scatter_(0, torch.tensor(tgt), 1)  # (tgt_subsampled_points)
        tgt = torch.sort(torch.tensor(tgt))[0]
        return pointcloud1[src, :], mask_src, pointcloud1[tgt, :], mask_tgt


def farthest_avg_subsample_points(point, npoint):
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point
