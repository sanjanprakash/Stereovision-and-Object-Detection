import numpy as np
import cv2
from matplotlib import pyplot as plt
import PointCloudLibrary as pcl
import StereoVisionLibrary as svl

IMG_left = cv2.imread('images/left.png')
IMG_right = cv2.imread('images/right.png')

#Finding disparity map
disparity = svl.compute_disparity(IMG_left, IMG_right)


plt.imshow(disparity)
plt.show()



