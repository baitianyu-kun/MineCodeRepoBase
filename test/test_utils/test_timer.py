import torch
import models.DGCNN.dgcnn as dgcnn
from utils.timer import Timer

model_time=Timer()

x = torch.rand((10, 1024, 3)).cuda()
model_time.tic()
dgcnn = dgcnn.DGCNN().cuda()
y = dgcnn(x)
model_time.toc()
print(y)
print(model_time.total_time)
print(model_time.avg)