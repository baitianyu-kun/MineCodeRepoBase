import torch

a = torch.arange(3)
b = torch.arange(3,6)  # [3, 4, 5]
print(a)
print(b)
print(torch.einsum('i,i->', [a, b]))
