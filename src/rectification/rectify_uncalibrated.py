import cv2
import numpy as np
# from matplotlib import pyplot as plt


img_1 = cv2.imread('left_1.png',1)
img_2 = cv2.imread('right_1.png',1)
rows,cols = img_1.shape[:2]
color = (0,255,0)

sift = cv2.SIFT()

camera_matrix_1 = np.matrix([[734.5626,0,284.7837],[0,797.2962,302.3755],[0,0,1]])#30.0783

camera_matrix_2 = np.matrix([[743.5223,0,333.2208],[0,802.5812,306.2985],[0,0,1]])#46.5610

dist_coeffs_1 = np.matrix([-0.1666,    0.1677,0,0,0])
dist_coeffs_2 = np.matrix([-0.2210,    0.8222,  0,0 ,0])

r = np.matrix([[0.9981,-0.0039,0.0615],[0.0034,1.00,0.0091],[-0.0616,-0.0088,0.9981]])
t = np.array([-62.2974,2.7549,4.8679])


( r_1,r_2,p_1,p_2,depth,roi_1,roi_2  ) = cv2.stereoRectify(camera_matrix_1,dist_coeffs_1,camera_matrix_2,dist_coeffs_2,(512,512),r,t,flags=0)


########################undistortion starts here ##########################
newcameramtx_1, roi_1=cv2.getOptimalNewCameraMatrix(camera_matrix_1,dist_coeffs_1,(cols,rows),1,(cols,rows))

img_l_undistort  = cv2.undistort(img_1, camera_matrix_1, dist_coeffs_1, None, newcameramtx_1)

newcameramtx_2, roi_2=cv2.getOptimalNewCameraMatrix(camera_matrix_2,dist_coeffs_2,(cols,rows),1,(cols,rows))

img_r_undistort  = cv2.undistort(img_2, camera_matrix_2, dist_coeffs_2, None, newcameramtx_2)


#####################rectification using sift feature #######################

kp1 = sift.detect(img_l_undistort,None)
kp2 = sift.detect(img_r_undistort,None)

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.compute(img_l_undistort,kp1)
kp2, des2 = sift.compute(img_r_undistort,kp2)
 
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
 
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)
 
good = []
pts1 = []
pts2 = []
 
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
	if m.distance < 0.8*n.distance:
	    good.append(m)
	    pts2.append(kp2[m.trainIdx].pt)
	    pts1.append(kp1[m.queryIdx].pt)

pts1 = np.float32(pts1)
pts2 = np.float32(pts2) 
# print pts1.dtype
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC )

points1r = pts1.reshape((pts1.shape[0] * 2, 1))
points2r = pts2.reshape((pts2.shape[0] * 2, 1))

ret, h_1,h_2 = cv2.stereoRectifyUncalibrated(points1r,points2r,F,(512,512))
img_l = np.array((512,512),np.float32)
img_r = np.array((512,512),np.float32)

img_l = cv2.warpPerspective(img_l_undistort,h_1,(512,512),img_l,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,0)
img_r = cv2.warpPerspective(img_r_undistort,h_2,(512,512),img_r,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,0)

###################lines for comparision it is optional ##########################
cv2.line(img_l, (0,100), (511,100), color,1)
cv2.line(img_l, (0,256), (511,256), color,1)
cv2.line(img_r, (0,100), (511,100), color,1)
cv2.line(img_r, (0,256), (511,256), color,1)
cv2.line(img_l, (0,300), (511,300), color,1)
cv2.line(img_l, (0,400), (511,400), color,1)
cv2.line(img_r, (0,300), (511,300), color,1)
cv2.line(img_r, (0,400), (511,400), color,1)






#####################stereo matching ##############################
# stereo = cv2.createStereoBM(numDisparities=16, blockSize=15)
minDisparity = -10
num_disparities = 80
sad_window = 9
# stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET,ndisparities=16, SADWindowSize=15)
# stereo = cv2.StereoSGBM(minDisparity,num_disparities,sad_window)
# disparity = stereo.compute(img_1,img_2)

# cv2.imshow('disparity',disparity)
# cv2.imwrite('left_7.png',img_l)
# cv2.imwrite('right_7.png',img_r)
cv2.imshow('left',img_l)
cv2.imshow('right',img_r)
cv2.waitKey(0)

