import torch
import torch.nn as nn
from sklearn.metrics import recall_score, precision_score, f1_score
import se_math.se3 as se3
import se_math.so3 as so3
import utils.data_utils as du
import numpy as np


def classificationloss(pred, gt_labels):
    """  Classification Loss for the inlier confidence
    Inputs:
        - pred: [bs, num_corr] predicted logits/labels for the putative correspondences
        - gt_labels:   [bs, num_corr] ground truth labels
    Outputs:(dict)
        - loss          BCE loss for inlier confidence
        - precision:    inlier precision (# kept inliers / # kepts matches)
        - recall:       inlier recall (# kept inliers / # all inliers)
        - f1:           (precision * recall * 2) / (precision + recall)
    """
    # CHANGED ATTENTION ORIGIN CODE 'BALANCED' PARAM
    loss = nn.BCEWithLogitsLoss(reduction='mean')(pred, gt_labels.float())
    # True -> int: if pred is logits then put it into labels
    pred_labels = (pred > 0).float()
    precision = precision_score(gt_labels[0], pred_labels[0])
    recall = recall_score(gt_labels[0], pred_labels[0])
    f1 = f1_score(gt_labels[0], pred_labels[0])
    return loss, precision, recall, f1


def spectralmatchingloss(M, gt_labels):
    """ Spectral Matching Loss
    Inputs:
        - M:    [bs, num_corr, num_corr] feature similarity matrix
        - gt_labels:   [bs, num_corr] ground truth inlier/outlier labels
    Output:
        - loss
    """
    batch, num_cors = gt_labels.shape
    gt_M = ((gt_labels[:, None, :] + gt_labels[:, :, None]) == 2).float()
    # set diagonal of gt_M to zero
    gt_M[:, torch.arange(num_cors), torch.arange(num_cors)] = 0
    loss = nn.MSELoss(reduction='mean')(M, gt_M)
    return loss


def transformationloss_origin(trans, gt_trans, src_keypts, tgt_keypts, probs, re_thre=15, te_thre=0.3):
    bs = trans.shape[0]
    R, t = se3.decompose_trans(trans)
    gt_R, gt_t = se3.decompose_trans(gt_trans)

    recall = 0
    RE = torch.tensor(0.0).to(trans.device)
    TE = torch.tensor(0.0).to(trans.device)
    RMSE = torch.tensor(0.0).to(trans.device)
    loss = torch.tensor(0.0).to(trans.device)
    for i in range(bs):
        re = torch.acos(torch.clamp((torch.trace(R[i].T @ gt_R[i]) - 1) / 2.0, min=-1, max=1))
        te = torch.sqrt(torch.sum((t[i] - gt_t[i]) ** 2))
        warp_src_keypts = se3.transform_torch(trans[i], src_keypts[i])
        rmse = torch.norm(warp_src_keypts - tgt_keypts, dim=-1).mean()
        re = re * 180 / np.pi
        te = te * 100
        if te < te_thre and re < re_thre:
            recall += 1
        RE += re
        TE += te
        RMSE += rmse

        pred_inliers = torch.where(probs[i] > 0)[0]
        if len(pred_inliers) < 1:
            loss += torch.tensor(0.0).to(trans.device)
        else:
            warp_src_keypts = se3.transform_torch(trans[i], src_keypts[i])
            loss += ((warp_src_keypts - tgt_keypts) ** 2).sum(-1).mean()

    return loss / bs, recall * 100.0 / bs, RE / bs, TE / bs, RMSE / bs


def transformationloss(trans, gt_trans, src_keypts, tgt_keypts, probs, re_thre=15, te_thre=0.3):
    """ Transformation Loss
    Inputs:
        - trans:      [bs, 4, 4] SE3 transformation matrices
        - gt_trans:   [bs, 4, 4] ground truth SE3 transformation matrices
        - src_keypts: [bs, num_corr, 3]
        - tgt_keypts: [bs, num_corr, 3]
        - probs:     [bs, num_corr] predicted inlier probability
    Outputs:
        - loss     transformation loss
        - recall   registration recall (re < re_thre & te < te_thre)
        - RE       rotation error
        - TE       translation error
        - RMSE     under the predicted transformation
    """
    batch_size, num_cors, _ = src_keypts.shape
    inv_trans = se3.inverse(trans)
    # R[0].T@R_gt[0] = (R[0]^-1)@R_gt[0] R is 正交矩阵, so R.T = R ^ 1
    concated_trans_gt_trans = se3.concatenate(inv_trans, gt_trans, device=trans.device)
    re = so3.RG2angle(concated_trans_gt_trans)
    R, t = se3.decompose_trans(trans)
    R_gt, t_Gt = se3.decompose_trans(gt_trans)
    te = torch.mean((t_Gt - t) ** 2, dim=1)
    src_keypts_trans = se3.transform_torch(trans.to(torch.float), src_keypts)
    rmse = torch.norm(src_keypts_trans - tgt_keypts, dim=-1).mean()
    recall = 0
    for i in range(batch_size):
        if re[i] < re_thre and te[i] < te_thre:
            recall += 1
    # RPMNet Eq 10
    loss = ((src_keypts_trans - tgt_keypts) ** 2).sum(-1).mean()
    return loss / batch_size, recall * 100 / batch_size, re.sum(-1) / batch_size, te.flatten().sum(
        -1) / batch_size, rmse / batch_size


if __name__ == '__main__':
    # pred = torch.rand((2, 1024))
    # gt_labels = torch.randint(0, 5, (2, 1024))
    # M = torch.rand((2, 1024, 1024))
    # spectralmatchingloss(M, gt_labels)
    # print(classificationloss(pred, gt_labels)

    trans_2batch = torch.concat([torch.from_numpy(du.random_pose(60, 0.1)).unsqueeze(0),
                                 torch.from_numpy(du.random_pose(60, 0.1)).unsqueeze(0)])
    gt_trans_2batch = torch.concat([torch.from_numpy(du.random_pose(60, 0.1)).unsqueeze(0),
                                    torch.from_numpy(du.random_pose(60, 0.1)).unsqueeze(0)])
    src_keypts = torch.rand((2, 5, 3))
    tgt_keypts = torch.rand((2, 5, 3))
    prob = torch.rand((2, 5))
    print(
        transformationloss(trans_2batch.to(torch.float), gt_trans_2batch.to(torch.float), src_keypts, tgt_keypts, prob))
    print(
        transformationloss_origin(trans_2batch.to(torch.float), gt_trans_2batch.to(torch.float), src_keypts, tgt_keypts,
                                  prob))
