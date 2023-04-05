import torch
import math

src_keypts = torch.rand((1, 5, 3))
src_keypts_trans = torch.rand((1, 5, 3))

# torch.norm 公式
print(torch.norm(src_keypts_trans - src_keypts))
print(torch.sqrt(((src_keypts_trans-src_keypts)**2).sum()))

# torch.mseloss 公式
print((((src_keypts_trans-src_keypts)**2).sum())/(src_keypts.shape[1]*src_keypts.shape[2]))
print((((src_keypts_trans-src_keypts)**2).sum(-1)))
print((((src_keypts_trans-src_keypts)**2)))
print(torch.nn.MSELoss()(src_keypts_trans, src_keypts))

# torch.norm -> torch.mse
print(((torch.norm(src_keypts_trans - src_keypts))**2)/(src_keypts.shape[1]*src_keypts.shape[2]))
