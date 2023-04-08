import numpy as np
import utils.visual as vi

data1=np.loadtxt('../data/airplane_0010.txt',delimiter=',')[:1024,:3]
data2=np.load('../data/airplane_0010_fcgf_down.npy')
print(data2.shape)
vi.show(data2)
