import numpy as np
import torch
from torch import nn
from se_math import so3
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
