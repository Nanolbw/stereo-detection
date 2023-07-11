import cv2
import numpy as np

# 左相机内参
left_camera_matrix = np.array([[416.841180253704, 0.0, 338.485167779639],
                                         [0., 416.465934495134, 230.419201769346],
                                         [0., 0., 1.]])

# 左相机畸变系数:[k1, k2, p1, p2, k3]
left_distortion = np.array([[-0.0170280933781798, 0.0643596519467521, -0.00161785356900972, -0.00330684695473645, 0]])

# 右相机内参
right_camera_matrix = np.array([[417.765094485395, 0.0, 315.061245379892],
                                          [0., 417.845058291483, 238.181766936442],
                                            [0., 0., 1.]])
# 右相机畸变系数:[k1, k2, p1, p2, k3]
right_distortion = np.array([[-0.0394089328586398, 0.131112076868352, -0.00133793245429668, -0.00188957913931929, 0]])

# om = np.array([-0.00009, 0.02300, -0.00372])
# R = cv2.Rodrigues(om)[0]

# 旋转矩阵
R = np.array([[0.999962872853149, 0.00187779299260463, -0.00840992323112715],
                           [ -0.0018408858041373, 0.999988651353238, 0.00439412154902114],
                           [ 0.00841807904053251, -0.00437847669953504, 0.999954981430194]])

# 平移向量
T = np.array([[-120.326603502087], [0.199732192805711], [-0.203594457929446]])

size = (640, 480)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)