import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from se_math import se3


class LossFunction(nn.Module):
    def __init__(self, loss_margin):
        super(LossFunction, self).__init__()
        self.criterion2 = ChamferLoss()
        self.criterion = nn.MSELoss(reduction='sum')
        self.GAL = GlobalAlignLoss()
        self.margin = loss_margin

    def forward(self, *input):
        """
        Compute global alignment loss and neighorhood consensus loss
        Args:
            src_keypoints: Keypoints of source point clouds. Size (B, 3, num_keypoint)
            tgt_keypoints: Keypoints of target point clouds. Size (B, 3, num_keypoint)
            rotation_ab: Size (B, 3, 3)
            translation_ab: Size (B, 3)
            src_keypoints_knn: [b, 3, num_kepoints, k]
            tgt_keypoints_knn: [b, 3, num_kepoints, k]
            k: Number of nearest neighbors.
            src_transformed: Transformed source point clouds. Size (B, 3, N)
            tgt: Target point clouds. Size (B, 3, M)
        Returns:
            neighborhood_consensus_loss
            global_alignment_loss
        """
        src_keypoints = input[0]
        tgt_keypoints = input[1]
        rotation_ab = input[2]
        translation_ab = input[3]
        src_keypoints_knn = input[4]
        tgt_keypoints_knn = input[5]
        k = input[6]
        src_transformed = input[7]
        tgt = input[8]
        batch_size = src_keypoints.size()[0]
        global_alignment_loss = self.GAL(src_transformed.permute(0, 2, 1), tgt.permute(0, 2, 1), self.margin)
        transformed_srckps_forward = se3.transform_torch(se3.integrate_trans(rotation_ab, translation_ab),
                                                         src_keypoints.transpose(1, 2)).transpose(1, 2)
        keypoints_loss = self.criterion(transformed_srckps_forward, tgt_keypoints)
        knn_consensus_loss = self.criterion(src_keypoints_knn, tgt_keypoints_knn)
        neighborhood_consensus_loss = knn_consensus_loss / k + keypoints_loss
        return neighborhood_consensus_loss, global_alignment_loss


class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)  # [b,n]
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x).to(x.device)
        diag_ind_y = torch.arange(0, num_points_y).to(x.device)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P


class GlobalAlignLoss(nn.Module):

    def __init__(self):
        super(GlobalAlignLoss, self).__init__()

    def forward(self, preds, gts, c):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        mins = self.huber_loss(mins, c)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        mins = self.huber_loss(mins, c)
        loss_2 = torch.sum(mins)

        return loss_1 + loss_2

    def huber_loss(self, x, c):
        x = torch.where(x < c, 0.5 * (x ** 2), c * x - 0.5 * (c ** 2))
        return x

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x).to(x.device)
        diag_ind_y = torch.arange(0, num_points_y).to(x.device)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P
