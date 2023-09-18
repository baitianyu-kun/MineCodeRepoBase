import numpy as np
import torch

# (1024, 3)
points = np.loadtxt('airplane_0627.txt', delimiter=',')[:1024, :3]
points = torch.from_numpy(points)
# (64, 3)
center = np.load('center.npy')[0]
# (64,)
fpx_idx = np.load('fpx_idx.npy')[0]
# (64, 32, 3)
neighbor = np.load('neighbor.npy')[0]
neighbor = torch.from_numpy(neighbor)
neighbor_origin=neighbor
# (2048,)
idx2 = np.load('idx2.npy')
# (819, 3)
ps = np.load('ps.npy')
# (819, 3)
pt = np.load('pt.npy')
# (1024,)
mask_src = np.load('mask_src.npy')
# (1024,)
mask_tgt = np.load('mask_tgt.npy')

# 64, 32, 1
# 这个是mask_src的neighbors，只留下ps的补充的那部分呢
# print(torch.from_numpy(mask_src[idx2]).view(64,32,-1).shape)

# 这个是ps与原points互补的部分，只需要重建这部分就完了
mask_src = torch.from_numpy(mask_src)
mask_src = mask_src.to(torch.bool)
mask_src_inverse = ~mask_src

# 需要重建部分的neighbor
neighbor = neighbor.view(-1, 3)
neighbor_need_rebuild = neighbor[mask_src_inverse[idx2]]
# 需要重建部分的center
center_need_rebuild = fpx_idx[mask_src_inverse[fpx_idx]]
print(center_need_rebuild)
# print(neighbor_origin.view(-1,3)[center_need_rebuild])
