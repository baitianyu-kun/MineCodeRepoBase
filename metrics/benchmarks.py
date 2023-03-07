import numpy as np
import torch
from torch import nn
from se_math import so3
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def evaluate_mask(mask, mask_gt):
    """ evaluate mask
    Args:
        mask: estimate mask
        mask_gt: ground truth mask
    Returns: accuracy, precision, recall, f1
    """
    accs = []
    preciss = []
    recalls = []
    f1s = []
    for m, m_gt in zip(mask, mask_gt):
        m = m.cpu()
        m_gt = m_gt.cpu()
        # mask, mask_gt: n维
        acc = accuracy_score(m_gt, m.detach())
        precis = precision_score(m_gt, m.detach(), zero_division=0)
        recall = recall_score(m_gt, m.detach(), zero_division=0)
        f1 = f1_score(m_gt, m.detach())
        accs.append(acc)
        preciss.append(precis)
        recalls.append(recall)
        f1s.append(f1)
    acc = np.mean(accs)
    precis = np.mean(preciss)
    recall = np.mean(recalls)
    f1 = np.mean(f1s)

    return acc, precis, recall, f1
