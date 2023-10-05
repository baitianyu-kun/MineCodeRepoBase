import numpy as np
import torch

from benchmarks_RRE_RTE import *
from benchmarks_RRE_RTE_zy import *

if __name__ == '__main__':
    # gt_rotations, rotations, gt_translations, translations, reg_success_thresh_rot,
    #  reg_success_thresh_trans
    gt_rotations = torch.rand((2, 3, 3))
    rotations = torch.rand((2, 3, 3))
    gt_translations = torch.rand((2, 3, 1))
    translations = torch.rand((2, 3, 1))
    reg_success_thresh_rot = 1
    reg_success_thresh_trans = 1

    src_points = torch.rand((2, 512, 3))

    rre, rte, recall = registration_recall(gt_rotations, rotations, gt_translations, translations,
                                           reg_success_thresh_rot, reg_success_thresh_trans)

    recall2 = compute_registration_recall(src_points, se3.integrate_trans(rotations, translations),
                                          se3.integrate_trans(gt_rotations, gt_translations),acceptance_rmse=1)
    print(recall)
    print(recall2)
