import se_math.se3 as se3
import se_math.quaternions as qu
import utils.data_utils as du
import numpy as np
from scipy.spatial.transform import Rotation

# pose = du.random_pose(45, 0.5)
# x, y, z, w = se3.mat2quat(pose)
# print(se3.quat2mat((x, y, z, w)))
# print(Rotation.from_quat([x, y, z, w]).as_matrix())
# print(pose)


pose1 = du.random_pose(45, 0.5)
pose2 = du.random_pose(45, 0.5)
pose = np.concatenate([pose1, pose2], axis=0).reshape((2, 4, 4))
R, t = se3.decompose_trans(pose)
quat=qu.rotmat2quat(R)
print(qu.quat2rotmat(quat))
print(R)
