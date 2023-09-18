import numpy as np
import torch

# (1024, 3)
points = np.loadtxt('airplane_0627.txt', delimiter=',')[:1024, :3]
points = torch.from_numpy(points).unsqueeze(0)
points=torch.cat([points,points,points])
batch,num_pts,_=points.shape
# torch.Size([1, 64])
fpx_idx = np.load('fpx_idx.npy')
fpx_idx = torch.from_numpy(fpx_idx).to(torch.long)
fpx_idx=torch.cat([fpx_idx,fpx_idx,fpx_idx])
mask_src = np.load('mask_src.npy')
mask_src = torch.from_numpy(mask_src).unsqueeze(0)
mask_src=torch.cat([mask_src,mask_src,mask_src])
mask_src = mask_src.to(torch.bool)
# torch.Size([1, 1024])
mask_src_inverse = ~mask_src
# 需要重建的center idx
fpx_idx_rebuild=fpx_idx[mask_src_inverse.view(-1)[fpx_idx]]
# print(points.view(-1,3)[fpx_idx_rebuild].view(batch,-1,3).shape)

print(points[mask_src_inverse].view(batch,-1,3).shape)
print(points[mask_src].view(batch,-1,3).shape)