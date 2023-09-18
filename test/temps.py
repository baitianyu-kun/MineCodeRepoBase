import visual as vi
import numpy as np


a0=np.load('/mnt/d/linshikeshihua/anchor_points0.npy')[0]
a1=np.load('/mnt/d/linshikeshihua/anchor_points1.npy')[0]
n0=np.load('/mnt/d/linshikeshihua/nooverlap_points0.npy')[0]
n1=np.load('/mnt/d/linshikeshihua/nooverlap_points1.npy')[0]
vi.show(a0,a1)
vi.show(n0,n1)