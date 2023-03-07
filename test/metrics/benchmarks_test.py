import numpy as np
import torch

from metrics import benchmarks
data = np.loadtxt('../data/airplane_0010.txt', delimiter=',')[:, 0:3]
data2 = np.loadtxt('../data/airplane_0627.txt', delimiter=',')[:, 0:3]
data2+=0.002
data = torch.from_numpy(data).unsqueeze(0)
data2 = torch.from_numpy(data2).unsqueeze(0)

