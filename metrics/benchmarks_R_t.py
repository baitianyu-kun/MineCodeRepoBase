import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from se_math import so3


# p template -> gtR -> p source
# but model output p source -> R -> p template
# So R == inverse(gtR)
# So pre: R, gt: inverse (gtR)
# So pre: G, gt: inverse (gtG)
def compute_R_t_metrics(R, t, gtR, gtt):
    inv_R, inv_t = inv_R_t(gtR, gtt)
    cur_r_mse, cur_r_mae = anisotropic_R_error(R, inv_R)
    cur_t_mse, cur_t_mae = anisotropic_t_error(t, inv_t)
    cur_r_isotropic = isotropic_R_error(R, inv_R)
    cur_t_isotropic = isotropic_t_error(t, inv_t, inv_R)
    return cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
           cur_t_isotropic


def summary_R_t_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic):
    r_mse = np.concatenate(r_mse, axis=0)
    r_mae = np.concatenate(r_mae, axis=0)
    t_mse = np.concatenate(t_mse, axis=0)
    t_mae = np.concatenate(t_mae, axis=0)
    r_isotropic = np.concatenate(r_isotropic, axis=0)
    t_isotropic = np.concatenate(t_isotropic, axis=0)

    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        np.sqrt(np.mean(r_mse)), np.mean(r_mae), np.sqrt(np.mean(t_mse)), \
        np.mean(t_mae), np.mean(r_isotropic), np.mean(t_isotropic)
    return r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic


def inv_R_t(R, t):
    inv_R = R.permute(0, 2, 1).contiguous()
    inv_t = - inv_R @ t[..., None]
    return inv_R, torch.squeeze(inv_t, -1)


def anisotropic_R_error(r1, r2, seq='xyz', degrees=True):
    """ Calculate mse, mae euler agnle error.
    Args:
        r1: shape=(B, 3, 3), pred
        r2: shape=(B, 3, 3), gt
        seq:
        degrees:
    Returns: r_mse, r_mae
    """
    if isinstance(r1, torch.Tensor):
        r1 = r1.cpu().detach().numpy()
    if isinstance(r2, torch.Tensor):
        r2 = r2.cpu().detach().numpy()
    assert r1.shape == r2.shape
    eulers1, eulers2 = [], []
    for i in range(r1.shape[0]):
        euler1 = Rotation.from_matrix(r1[i]).as_euler(seq=seq, degrees=degrees)
        euler2 = Rotation.from_matrix(r2[i]).as_euler(seq=seq, degrees=degrees)
        eulers1.append(euler1)
        eulers2.append(euler2)
    eulers1 = np.stack(eulers1, axis=0)
    eulers2 = np.stack(eulers2, axis=0)
    r_mse = np.mean((eulers1 - eulers2) ** 2, axis=-1)
    r_mae = np.mean(np.abs(eulers1 - eulers2), axis=-1)
    return r_mse, r_mae


def anisotropic_t_error(t1, t2):
    """ calculate translation mse and mae error.
    Args:
        t1: shape=(B, 3)
        t2: shape=(B, 3)
    Returns:
    """
    if isinstance(t1, torch.Tensor):
        t1 = t1.cpu().detach().numpy()
    if isinstance(t2, torch.Tensor):
        t2 = t2.cpu().detach().numpy()
    assert t1.shape == t2.shape
    t_mse = np.mean((t1 - t2) ** 2, axis=1)
    t_mae = np.mean(np.abs(t1 - t2), axis=1)
    return t_mse, t_mae


def isotropic_R_error(r1, r2):
    """ Calculate isotropic rotation degree error between r1 and r2.
    Args:
        r1: shape=(B, 3, 3), pred
        r2: shape=(B, 3, 3), gt
    Returns:
    """
    r2_inv = r2.permute(0, 2, 1).contiguous()
    r1r2 = torch.matmul(r2_inv, r1)
    # device = r1.device
    # B = r1.shape[0]
    # mask = torch.unsqueeze(torch.eye(3).to(device), dim=0).repeat(B, 1, 1)
    # tr = torch.sum(torch.reshape(mask * r1r2, (B, 9)), dim=-1)
    tr = r1r2[:, 0, 0] + r1r2[:, 1, 1] + r1r2[:, 2, 2]
    rads = torch.acos(torch.clamp((tr - 1) / 2, -1, 1))
    degrees = rads / math.pi * 180
    return degrees


def isotropic_t_error(t1, t2, R2):
    """ Calculate isotropic translation error between t1 and t2.
    Args:
        t1: shape=(B, 3), pred_t
        t2: shape=(B, 3), gtt
        R2: shape=(B, 3, 3), gtR
    Returns:
    """
    R2, t2 = inv_R_t(R2, t2)
    error = torch.squeeze(R2 @ t1[..., None], -1) + t2
    error = torch.norm(error, dim=-1)
    return error


def compute_te(t_est, t):
    """Computes the translation error.
    Modified from PCAM source code https://github.com/valeoai/PCAM
    """
    return np.linalg.norm(t - t_est)


def compute_re(R_est, R):
    """Computes the rotation error in degrees
    Modified from PCAM source code https://github.com/valeoai/PCAM
    """
    eps = 1e-16
    return np.arccos(
        np.clip(
            (np.trace(R_est.T @ R) - 1) / 2,
            -1 + eps,
            1 - eps
        )
    ) * 180. / np.pi


def compute_rmse_mae(R_est, R):
    """ compute RMSE and RMAE
    Args:
        R_est: R estimated
        R: R ground truth
    Returns:
    """
    if isinstance(R_est, torch.Tensor):
        R_est = R_est.cpu().detach().numpy()
    if isinstance(R, torch.Tensor):
        R = R.cpu().detach().numpy()
    r_est_deg = so3.dcm2euler(R_est, seq='xyz')
    r_deg = so3.dcm2euler(R, seq='xyz')
    r_mse = np.mean((r_deg - r_est_deg) ** 2, axis=1)
    r_mae = np.mean(np.abs(r_deg - r_est_deg), axis=1)
    return r_mse, r_mae


def compute_tmse_mae(t_est, t):
    """ compute tmse and tmae
    Args:
        t_est:
        t:
    Returns:
    """
    if isinstance(t_est, torch.Tensor):
        t_est = t_est.cpu().detach().numpy()
    if isinstance(t, torch.Tensor):
        t = t.cpu().detach().numpy()
    t_mse = np.mean((t_est - t) ** 2, axis=1)
    t_mae = np.mean(np.abs(t_est - t), axis=1)
    return np.mean(t_mse), np.mean(t_mae)
