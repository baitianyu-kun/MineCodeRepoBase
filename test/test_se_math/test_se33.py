import se_math.se3 as se3
import utils.data_utils as du

G = du.random_pose(60, 0.5)
se3.inverse_np(G)