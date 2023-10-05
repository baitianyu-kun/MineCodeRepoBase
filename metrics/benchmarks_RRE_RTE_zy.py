import torch
import numpy as np
import math


def relative_rotation_error(R2, R1):
    """
    Torch batch implementation of the rotation error between the estimated and the ground truth rotatiom matrix.
    Rotation error is defined as r_e = \arccos(\frac{Trace(\mathbf{R}_{ij}^{T}\mathbf{R}_{ij}^{\mathrm{GT}) - 1}{2})

    Args:
        R1 (torch tensor): Estimated rotation matrices [b,3,3]
        R2 (torch tensor): Ground truth rotation matrices [b,3,3]

    Returns:
        ae (torch tensor): Rotation error in angular degreees [b,1]

    """
    if isinstance(R1, np.ndarray):
        R1 = torch.from_numpy(R1)
    if isinstance(R2, np.ndarray):
        R2 = torch.from_numpy(R2)
    R_ = torch.matmul(R1.permute(0, 2, 1), R2)
    e = torch.stack([(torch.trace(R_[_, :, :]) - 1) / 2 for _ in range(R_.shape[0])], dim=0).unsqueeze(1)

    # Clamp the errors to the valid range (otherwise torch.acos() is nan)
    e = torch.clamp(e, -1, 1, out=None)

    ae = torch.acos(e)
    pi = torch.Tensor([math.pi])
    ae = 180. * ae / pi.to(ae.device).type(ae.dtype)

    return ae


def relative_translation_error(gt_translations, translations):
    r"""Isotropic Relative Rotation Error.
    RTE = \lVert t - \bar{t} \rVert_2
    Args:
        gt_translations (Tensor): ground truth translation vector (*, 3)
        translations (Tensor): estimated translation vector (*, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    if isinstance(translations, np.ndarray):
        translations = torch.from_numpy(translations)
    if isinstance(gt_translations, np.ndarray):
        gt_translations = torch.from_numpy(gt_translations)
    gt_translations = gt_translations.reshape(gt_translations.shape[0], 3)
    translations = translations.reshape(translations.shape[0], 3)
    rte = torch.linalg.norm(gt_translations - translations, dim=-1)
    return rte


def registration_recall(gt_rotations, rotations, gt_translations, translations, reg_success_thresh_rot,
                        reg_success_thresh_trans):
    rre = relative_rotation_error(gt_rotations, rotations)
    rte = relative_translation_error(gt_translations, translations)
    recall = torch.logical_and(torch.lt(rre, reg_success_thresh_rot),
                               torch.lt(rte, reg_success_thresh_trans)).float()
    rre = torch.mean(rre)
    rte = torch.mean(rte)
    recall = torch.mean(recall)
    return rre, rte, recall
