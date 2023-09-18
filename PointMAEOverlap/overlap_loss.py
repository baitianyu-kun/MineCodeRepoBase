import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import se_math.se3 as se3


def chamfer_loss(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    # diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    diag_ind = torch.arange(0, num_points)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return torch.mean(torch.min(P, 1)[0]) + torch.mean(torch.min(P, 2)[0])


def chamfer_loss_overlap(x, y, x_mask, y_mask):
    # only calculate the loss in the overlap region between x and y
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    # diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    diag_ind = torch.arange(0, num_points)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    # return torch.mean(x_mask * (torch.min(P, 1)[0])) + torch.mean(y_mask * (torch.min(P, 2)[0]))
    # 应该让这两个重叠区域的loss值相同，所以是不是得用减法，或者MSE loss
    # or (torch.mean(x_mask * (torch.min(P, 1)[0])) - torch.mean(y_mask * (torch.min(P, 2)[0])))**2
    return F.mse_loss(torch.mean(x_mask * (torch.min(P, 1)[0])), torch.mean(y_mask * (torch.min(P, 2)[0])))


class ChamferLossOverlap(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, x_mask, y_mask):
        # only calculate the loss in the overlap region between x and y
        bs, num_points, points_dim = x.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        # diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
        diag_ind = torch.arange(0, num_points)
        rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        # return torch.mean(x_mask * (torch.min(P, 1)[0])) + torch.mean(y_mask * (torch.min(P, 2)[0]))
        # 应该让这两个重叠区域的loss值相同，所以是不是得用减法，或者MSE loss
        # or (torch.mean(x_mask * (torch.min(P, 1)[0])) - torch.mean(y_mask * (torch.min(P, 2)[0])))**2
        return F.mse_loss(torch.mean(x_mask * (torch.min(P, 1)[0])), torch.mean(y_mask * (torch.min(P, 2)[0])))


if __name__ == '__main__':
    p1 = np.load('./overlap_data_test/p1.npy')
    p2 = np.load('./overlap_data_test/p2.npy')
    p2 += 0.02
    p1mask = np.load('./overlap_data_test/p1mask.npy')
    p2mask = np.load('./overlap_data_test/p2mask.npy')
    p1 = torch.from_numpy(p1).unsqueeze(0)
    p1 = torch.cat([p1, p1 + 0.03])
    p2 = torch.from_numpy(p2).unsqueeze(0)
    p2 = torch.cat([p2, p2])
    p1mask = torch.from_numpy(p1mask).unsqueeze(0)
    p1mask = torch.cat([p1mask, p1mask])
    p2mask = torch.from_numpy(p2mask).unsqueeze(0)
    p2mask = torch.cat([p2mask, p2mask])
    #
    cd_overlap = chamfer_loss_overlap(p1, p2, p1mask, p2mask)
    chamfer_loss2=ChamferLossOverlap()
    cd_overlap2=chamfer_loss2(p1,p2,p1mask,p2mask)
    print(cd_overlap,cd_overlap2)
