import numpy as np
import cv2
import camera_config
import random
import math
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 480)  #打开并设置摄像头
# 鼠标回调函数
def onmouse_pick_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        threeD = param
        print('\n像素坐标 x = %d, y = %d' % (x, y))
        # print("世界坐标是：", threeD[y][x][0], threeD[y][x][1], threeD[y][x][2], "mm")
        print("世界坐标xyz 是：", threeD[y][x][0]/ 1000.0 , threeD[y][x][1]/ 1000.0 , threeD[y][x][2]/ 1000.0 , "m")

        distance = math.sqrt( threeD[y][x][0] **2 + threeD[y][x][1] **2 + threeD[y][x][2] **2 )
        distance = distance / 1000.0  # mm -> m
        print("距离是：", distance, "m")

WIN_NAME = 'Deep disp'
cv2.namedWindow(WIN_NAME,  cv2.WINDOW_AUTOSIZE)

while True:
  ret, frame = cap.read()
  frame1 = frame[0:480, 0:640]
  frame2 = frame[0:480, 640:1280]  #割开双目图像

  imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片
  imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

  # cv2.remap 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
  # 依据MATLAB测量数据重建无畸变图片
  img1_rectified = cv2.remap(imgL, camera_config.left_map1, camera_config.left_map2, cv2.INTER_LINEAR)
  img2_rectified = cv2.remap(imgR, camera_config.right_map1, camera_config.right_map2, cv2.INTER_LINEAR)

  imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
  imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

  # BM
  numberOfDisparities = ((640 // 8) + 15) & -16  # 640对应是分辨率的宽

  stereo = cv2.StereoBM_create(numDisparities=16, blockSize=9)  #立体匹配
  stereo.setROI1(camera_config.validPixROI1)
  stereo.setROI2(camera_config.validPixROI2)
  stereo.setPreFilterCap(31)
  stereo.setBlockSize(15)
  stereo.setMinDisparity(0)
  stereo.setNumDisparities(numberOfDisparities)
  stereo.setTextureThreshold(10)
  stereo.setUniquenessRatio(15)
  stereo.setSpeckleWindowSize(100)
  stereo.setSpeckleRange(32)
  stereo.setDisp12MaxDiff(1)

  disparity = stereo.compute(img1_rectified, img2_rectified) # 计算视差

  disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  #归一化函数算法

  threeD = cv2.reprojectImageTo3D(disparity, camera_config.Q, handleMissingValues=True)  #计算三维坐标数据值
  threeD = threeD * 16

  # threeD[y][x] x:0~640; y:0~480;   !!!!!!!!!!
  cv2.setMouseCallback(WIN_NAME, onmouse_pick_points, threeD)

  cv2.imshow("left", frame1)
  # cv2.imshow("right", frame2)
  # cv2.imshow("left_r", imgL)
  # cv2.imshow("right_r", imgR)
  cv2.imshow(WIN_NAME, disp)  #显示深度图的双目画面

  key = cv2.waitKey(1)
  if key == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()

# 立体匹配
iml_, imr_ = preprocess(iml, imr)  # 预处理，一般可以削弱光照不均的影响，不做也可以
disp, _ = stereoMatchSGBM(iml, imr, False)  # 这里传入的是未经立体校正的图像，因为我们使用的middleburry图片已经是校正过的了
cv2.imwrite('./data/disaprity.png', disp * 4)
# 计算深度图
# depthMap = getDepthMapWithQ(disp, Q)
depthMap = getDepthMapWithConfig(disp, config)
print(depthMap)
minDepth = np.min(depthMap)
maxDepth = np.max(depthMap)
print(minDepth, maxDepth)
depthMapVis = (255.0 * (depthMap - minDepth)) / (maxDepth - minDepth)
depthMapVis = depthMapVis.astype(np.uint8)
cv2.imshow("DepthMap", depthMapVis)
cv2.waitKey(0)


def getDepthMapWithQ(disparityMap: np.ndarray, Q: np.ndarray) -> np.ndarray:
  points_3d = cv2.reprojectImageTo3D(disparityMap, Q)
  depthMap = points_3d[:, :, 2]
  reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
  depthMap[reset_index] = 0

  return depthMap.astype(np.float32)


def getDepthMapWithConfig(disparityMap: np.ndarray, config: stereoconfig.stereoCamera) -> np.ndarray:
  fb = config.cam_matrix_left[0, 0] * (-config.T[0])
  doffs = config.doffs
  depthMap = np.divide(fb, disparityMap + doffs)
  reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
  depthMap[reset_index] = 0
  reset_index2 = np.where(disparityMap < 0.0)
  depthMap[reset_index2] = 0
  return depthMap.astype(np.float32)