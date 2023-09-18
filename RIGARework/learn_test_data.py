import torch

from RIGARework.partial_threeDmatchDataset import ThreeDMatchPartial

ps, pt, iGgt, iRgt, itgt, gt_mask_src, gt_mask_tgt = ThreeDMatchPartial(
    DATA_DIR='D:\\dataset\\sun3d-home_at-home_at_scan1_2013_jan_1', partition='test',
    num_pts=10).__getitem__(0)

# print(ps)
# print(gt_mask_src)
# print(torch.where(gt_mask_src==1)[0])
# print(ps[gt_mask_src == 1, :])

batch=2
num_pts=ps.shape[0]

ps = torch.from_numpy(ps).unsqueeze(0).repeat(batch, 1, 1)
pt = torch.from_numpy(pt).unsqueeze(0).repeat(batch, 1, 1)

gt_mask_src = torch.Tensor(gt_mask_src).unsqueeze(0).repeat(batch,1)

# (batch, num_pts, 3)
gt_mask_src=gt_mask_src.repeat_interleave(3,dim=1).view(batch,num_pts,3)

print(ps)
print(gt_mask_src)
# 带batch的取重叠部分
print(ps[gt_mask_src==1].view(batch,-1,3))