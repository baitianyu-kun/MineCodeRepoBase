import torch
import torch.nn as nn
import numpy as np
from models.PointNet.pointnet import PointNet


pointnet = PointNet(emb_dims=1024, input_shape='bnc', use_bn=True, global_feat=False)
p1=torch.rand((2,1024,3))
print(pointnet(p1).shape)