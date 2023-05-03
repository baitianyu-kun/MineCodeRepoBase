import numpy as np
import cv2 as cv

match_mtx_ps = np.load('match_mtx_ps_gt.npy')
match_mtx_pt = np.load('match_mtx_pt_gt.npy')
print(match_mtx_pt)
cv.imshow('test', match_mtx_pt)
cv.waitKey(0)
cv.destroyAllWindows()
