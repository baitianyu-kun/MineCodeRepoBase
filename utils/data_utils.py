import math
import numpy as np
import torch
from scipy.spatial.distance import minkowski
from sklearn.neighbors import NearestNeighbors, KDTree
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
    """ add noise to pcd
    Args:
        pcd:
        sigma:
        clip:
    Returns:
    """
    pcd += np.clip(sigma * np.random.randn(*pcd.shape), -1 * clip, clip)
    return pcd


def make_patches(points, normals, k):
    """ Make points patches
    Args:
        points: pcds
        normals: pcds normals
        k: the k number of patches
    Returns: centers, centers_normals, patches, patches_normals, centers_idx
    """
    centers_idx = np.random.choice(len(points), size=len(points), replace=False)
    centers = points[centers_idx]
    kd_tree = KDTree(points)
    indexes = kd_tree.query(centers, k=k, return_distance=False)
    patches = points[indexes]
    centers_normals = normals[centers_idx]
    patches_normals = normals[indexes]
    return centers, centers_normals, patches, patches_normals, centers_idx


def add_outliers(pointcloud, gt_mask):
    """ Add outlier to pcds
    Args:
        pointcloud:
        gt_mask:
    Returns:
    """
    if isinstance(pointcloud, np.ndarray):
        pointcloud = torch.from_numpy(pointcloud)

    num_outliers = 20
    N, C = pointcloud.shape
    outliers = 2 * torch.rand(num_outliers, C) - 1  # Sample points in a cube [-0.5, 0.5]
    pointcloud = torch.cat([pointcloud, outliers], dim=0)
    gt_mask = torch.cat([gt_mask, torch.zeros(num_outliers)])

    idx = torch.randperm(pointcloud.shape[0])
    pointcloud, gt_mask = pointcloud[idx], gt_mask[idx]
    return pointcloud.numpy(), gt_mask


def compute_normal(pointcloud, radius=0.03, max_nn=30):
    """ Compute pcd normals
    Args:
        pointcloud:
        radius: 搜索半径
        max_nn: 邻域内用于估算法线的最大点数
    Returns:
    """
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(pointcloud)
    pcd_.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    normals = np.asarray(pcd_.normals)
    return normals.astype('float32')


def compute_ppf(centers, centers_normals, patches, patches_normals):
    centers = centers[:, np.newaxis, :]
    centers_normals = centers_normals[:, np.newaxis, :]
    delta = patches - centers
    dist = np.linalg.norm(delta, axis=-1)
    angle_0 = angle(centers_normals, delta)
    angle_1 = angle(patches_normals, delta)
    angle_2 = angle(centers_normals, patches_normals)
    ppf = np.stack((dist, angle_0, angle_1, angle_2), axis=-1)
    features = np.concatenate((delta, patches_normals, ppf), axis=-1)
    return features


def angle(u, v):
    """ Compute angle between 2 vectors
    For robustness, we use the same formulation as in PPFNet, i.e.
        angle(v1, v2) = atan2(cross(v1, v2), dot(v1, v2)).
    This handles the case where one of the vectors is 0.0, since torch.atan2(0.0, 0.0)=0.0
    Args:
        u:
        v:
    Returns:
    """
    cross = np.linalg.norm(np.cross(u, v), axis=-1)
    dot = np.sum(u * v, axis=-1)
    return np.arctan2(cross, dot)


def angle_torch(v1, v2):
    """ Compute angle between 2 vectors
    For robustness, we use the same formulation as in PPFNet, i.e.
        angle(v1, v2) = atan2(cross(v1, v2), dot(v1, v2)).
    This handles the case where one of the vectors is 0.0, since torch.atan2(0.0, 0.0)=0.0
    Args:
        v1: (B, *, 3)
        v2: (B, *, 3)
    Returns:
    """
    cross_prod = torch.stack([v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
                              v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
                              v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]], dim=-1)
    cross_prod_norm = torch.norm(cross_prod, dim=-1)
    dot_prod = torch.sum(v1 * v2, dim=-1)
    return torch.atan2(cross_prod_norm, dot_prod)


def make_patches(points, normals, k):
    centers_idx = np.random.choice(len(points), size=len(points), replace=False)
    centers = points[centers_idx]
    kd_tree = KDTree(points)
    indexes = kd_tree.query(centers, k=k, return_distance=False)
    patches = points[indexes]
    centers_normals = normals[centers_idx]
    patches_normals = normals[indexes]
    return centers, centers_normals, patches, patches_normals, centers_idx


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


def rotation_x_matrix(theta):
    mat = np.eye(3, dtype=np.float32)
    mat[1, 1] = math.cos(theta)
    mat[1, 2] = -math.sin(theta)
    mat[2, 1] = math.sin(theta)
    mat[2, 2] = math.cos(theta)
    return mat


def rotation_y_matrix(theta):
    mat = np.eye(3, dtype=np.float32)
    mat[0, 0] = math.cos(theta)
    mat[0, 2] = math.sin(theta)
    mat[2, 0] = -math.sin(theta)
    mat[2, 2] = math.cos(theta)
    return mat


def rotation_z_matrix(theta):
    mat = np.eye(3, dtype=np.float32)
    mat[0, 0] = math.cos(theta)
    mat[0, 1] = -math.sin(theta)
    mat[1, 0] = math.sin(theta)
    mat[1, 1] = math.cos(theta)
    return mat


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist
    return np.expand_dims(t, 1)


def random_select_points(pc, m):
    """ Random select points from pcd
    Args:
        pc: pcd points
        m:  num sampled
    Returns: sampled pcd points
    """
    if m < 0:
        idx = np.arange(pc.shape[0])
        np.random.shuffle(idx)
        return pc[idx, :]
    n = pc.shape[0]
    replace = False if n >= m else True
    idx = np.random.choice(n, size=(m,), replace=replace)
    return pc[idx, :]


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
