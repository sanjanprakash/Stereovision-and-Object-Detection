##################################################
# Filename: main.py
# Function:
#	1. reads the right and left image (from stereo camera)
#	2. finds disparity and depth map 
#	3. identifies object using grabcut() algorithm 
#	4. uses disparity to get a better segmentation result
#	5. displays the original images, disparity map, 
#	   depth map, detected object without using disparity, 
#	   detected object using disparity.
#	6. Finally shows the image with the detected object 
#	   and how far it is from the camera
##################################################

import cv2
import numpy as np 
import time
from matplotlib import pyplot as plt
import StereoVisionLibrary as svl
import ObjectDetectionLibrary as odl 

#Reading the images: 
IMG_left = cv2.imread('/home/radhika/IVP-Project/reimage4/left_25.png')
IMG_right = cv2.imread('/home/radhika/IVP-Project/reimage4/right_25.png')

#Finding disparity map
disparity = svl.compute_disparity(IMG_left, IMG_right)

#object detection without using disparity
IMG_detected_1 = odl.find_object_rect(IMG_left, IMG_right)

#object detection using disparity
IMG_detected_2 = odl.find_object_mask(IMG_left, IMG_right, disparity)
im2 = np.copy(IMG_detected_2)
#depth/distance of object
distance = svl.compute_depth(im2, disparity)
string = "distance = " + str(distance)
#display 

plt.figure(1)

plt.subplot(2,3,1)
plt.title("Left Image")
plt.imshow(IMG_left,'gray')

plt.subplot(2,3,2)
plt.title("Right Image")
plt.imshow(IMG_left,'gray')

plt.subplot(2,3,3)
plt.title("Disparity")
plt.imshow(disparity)

plt.subplot(2,3,4)
plt.title("obj without Disparity")
plt.imshow(IMG_detected_1,'gray')

plt.subplot(2,3,5)
plt.title("obj with Disparity")
plt.imshow(IMG_detected_2,'gray')

plt.subplot(2,3,6)
plt.title(string)
plt.imshow(im2,'gray')


plt.show()  





