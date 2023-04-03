import torch

data = torch.rand((2, 5, 5))
# [0,1,2],[0,1,2] -> (0,0) (1,1) (2,2)
data[:, [0, 1, 2], [0, 1, 2]] = 0
print(data)
