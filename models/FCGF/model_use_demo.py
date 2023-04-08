import torch
import model.misc as misc
import model
import numpy as np

data = np.loadtxt('../../test/data/airplane_0010.txt', delimiter=',')[:1024, :3]
Model = model.load_model('ResUNetBN2C')
# the in_channels has to be 1, see misc.etract_features func
model = Model(
    in_channels=1,
    out_channels=16,
    bn_momentum=0.1,
    normalize_feature=True,
    conv1_kernel_size=3,
    D=3
)
xyz_down, feature = misc.extract_features(
    model,
    xyz=data,
    voxel_size=0.025,
    device='cpu',
    skip_check=True
)
# and the feature should also sparse tensor -> tensor
# but it seems came back with torch.Tensor, so will change it later
print(feature.shape)
print(xyz_down.shape)
