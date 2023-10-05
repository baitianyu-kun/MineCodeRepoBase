import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

import se_math.se3 as se3


def relative_rotation_error(gt_rotations, rotations):
    r"""Isotropic Relative Rotation Error.
    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)
    Args:
        gt_rotations (Tensor): ground truth rotation matrix (*, 3, 3)
        rotations (Tensor): estimated rotation matrix (*, 3, 3)
    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    mat = torch.matmul(rotations.transpose(-1, -2), gt_rotations)
    trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    x = 0.5 * (trace - 1.0)
    x = x.clamp(min=-1.0, max=1.0)
    x = torch.arccos(x)
    rre = 180.0 * x / np.pi
    return rre


def relative_translation_error(gt_translations, translations):
    r"""Isotropic Relative Rotation Error.
    RTE = \lVert t - \bar{t} \rVert_2
    Args:
        gt_translations (Tensor): ground truth translation vector (*, 3)
        translations (Tensor): estimated translation vector (*, 3)
    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    rte = torch.linalg.norm(gt_translations - translations, dim=-1)
    return rte


def isotropic_transform_error(gt_transforms, transforms, reduction='mean'):
    r"""Compute the isotropic Relative Rotation Error and Relative Translation Error.
    Args:
        gt_transforms (Tensor): ground truth transformation matrix (*, 4, 4)
        transforms (Tensor): estimated transformation matrix (*, 4, 4)
        reduction (str='mean'): reduction method, 'mean', 'sum' or 'none'
    Returns:
        rre (Tensor): relative rotation error.
        rte (Tensor): relative translation error.
    """
    assert reduction in ['mean', 'sum', 'none']
    gt_rotations, gt_translations = se3.decompose_trans(gt_transforms)
    rotations, translations = se3.decompose_trans(transforms)
    rre = relative_rotation_error(gt_rotations, rotations)  # (*)
    rte = relative_translation_error(gt_translations, translations)  # (*)
    if reduction == 'mean':
        rre = rre.mean()
        rte = rte.mean()
    elif reduction == 'sum':
        rre = rre.sum()
        rte = rte.sum()
    return rre, rte


def anisotropic_transform_error(gt_transforms, transforms, reduction='mean'):
    r"""Compute the anisotropic Relative Rotation Error and Relative Translation Error.
    This function calls numpy-based implementation to achieve batch-wise computation and thus is non-differentiable.
    Args:
        gt_transforms (Tensor): ground truth transformation matrix (B, 4, 4)
        transforms (Tensor): estimated transformation matrix (B, 4, 4)
        reduction (str='mean'): reduction method, 'mean', 'sum' or 'none'
    Returns:
        r_mse (Tensor): rotation mse.
        r_mae (Tensor): rotation mae.
        t_mse (Tensor): translation mse.
        t_mae (Tensor): translation mae.
    """

    def compute_rotation_mse_and_mae(gt_rotation: np.ndarray, est_rotation: np.ndarray):
        r"""Compute anisotropic rotation error (MSE and MAE)."""
        gt_euler_angles = Rotation.from_matrix(gt_rotation).as_euler('xyz', degrees=True)  # (3,)
        est_euler_angles = Rotation.from_matrix(est_rotation).as_euler('xyz', degrees=True)  # (3,)
        mse = np.mean((gt_euler_angles - est_euler_angles) ** 2)
        mae = np.mean(np.abs(gt_euler_angles - est_euler_angles))
        return mse, mae

    def compute_translation_mse_and_mae(gt_translation: np.ndarray, est_translation: np.ndarray):
        r"""Compute anisotropic translation error (MSE and MAE)."""
        mse = np.mean((gt_translation - est_translation) ** 2)
        mae = np.mean(np.abs(gt_translation - est_translation))
        return mse, mae

    def compute_transform_mse_and_mae(gt_transform: np.ndarray, est_transform: np.ndarray):
        r"""Compute anisotropic rotation and translation error (MSE and MAE)."""
        gt_rotation, gt_translation = se3.decompose_trans(gt_transform)
        est_rotation, est_translation = se3.decompose_trans(est_transform)
        r_mse, r_mae = compute_rotation_mse_and_mae(gt_rotation, est_rotation)
        t_mse, t_mae = compute_translation_mse_and_mae(gt_translation, est_translation)
        return r_mse, r_mae, t_mse, t_mae

    assert reduction in ['mean', 'sum', 'none']
    batch_size = gt_transforms.shape[0]
    gt_transforms_array = gt_transforms.detach().cpu().numpy()
    transforms_array = transforms.detach().cpu().numpy()
    all_r_mse = []
    all_r_mae = []
    all_t_mse = []
    all_t_mae = []
    for i in range(batch_size):
        r_mse, r_mae, t_mse, t_mae = compute_transform_mse_and_mae(gt_transforms_array[i], transforms_array[i])
        all_r_mse.append(r_mse)
        all_r_mae.append(r_mae)
        all_t_mse.append(t_mse)
        all_t_mae.append(t_mae)
    r_mse = torch.as_tensor(all_r_mse).to(gt_transforms)
    r_mae = torch.as_tensor(all_r_mae).to(gt_transforms)
    t_mse = torch.as_tensor(all_t_mse).to(gt_transforms)
    t_mae = torch.as_tensor(all_t_mae).to(gt_transforms)
    if reduction == 'mean':
        r_mse = r_mse.mean()
        r_mae = r_mae.mean()
        t_mse = t_mse.mean()
        t_mae = t_mae.mean()
    elif reduction == 'sum':
        r_mse = r_mse.sum()
        r_mae = r_mae.sum()
        t_mse = t_mse.sum()
        t_mae = t_mae.sum()
    return r_mse, r_mae, t_mse, t_mae


def compute_inlier_ratio(ref_corr_points, src_corr_points, transform, positive_radius=0.1):
    r"""Computing the inlier ratio between a set of correspondences."""
    src_corr_points = se3.transform_np(transform, src_corr_points)
    residuals = np.sqrt(((ref_corr_points - src_corr_points) ** 2).sum(1))
    inlier_ratio = np.mean(residuals < positive_radius)
    return inlier_ratio


def get_nearest_neighbor(
        q_points: np.ndarray,
        s_points: np.ndarray,
        return_index: bool = False,
):
    r"""Compute the nearest neighbor for the query points in support points."""
    s_tree = cKDTree(s_points)
    distances, indices = s_tree.query(q_points, k=1, n_jobs=-1)
    if return_index:
        return distances, indices
    else:
        return distances


def compute_overlap(ref_points, src_points, transform=None, positive_radius=0.1):
    r"""Compute the overlap of two point clouds."""
    if transform is not None:
        src_points = se3.transform_np(transform, src_points)
    nn_distances = get_nearest_neighbor(ref_points, src_points)
    overlap = np.mean(nn_distances < positive_radius)
    return overlap


def compute_registration_recall(src_points, transform, gt_transform, acceptance_rmse=0.2):
    r"""Compute the registration recall"""
    transform_realignment = torch.matmul(torch.inverse(gt_transform), transform)
    src_points_realigned = se3.transform_torch(transform_realignment, src_points)
    rmse = torch.linalg.norm(src_points_realigned - src_points, dim=1).mean()
    recall = torch.lt(rmse, acceptance_rmse).float()
    return recall