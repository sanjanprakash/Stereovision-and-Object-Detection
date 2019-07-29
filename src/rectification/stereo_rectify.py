import cv2
import numpy as np
from matplotlib import pyplot as plt


img_1 = cv2.imread('left_5.png',1)
img_2 = cv2.imread('right_5.png',1)
rows,cols = img_1.shape[:2]

'''camera_matrix_1 = np.matrix([[6.431071007716022e+02,0,294.6248],[0,1.177128650603916e+03,5.146612567771655e+02],[0,0,1]])#30.0783

camera_matrix_2 = (1.0e+03)*np.matrix([[0.6537,0,0.3020],[0,1.1825,0.5609],[0,0,0.0010]])#46.5610

dist_coeffs_1 = np.matrix([-0.1702,    0.4512,0,0,0])
dist_coeffs_2 = np.matrix([-0.1430,    0.1704,  0,0 ,0])

r = np.matrix([[0.9997,0.0198,0.0149],[-0.0203,0.9992,0.0354],[-0.0142,-0.0357,0.9993]])
t = np.array([-88.0120,-15.4548,52.2919])'''


camera_matrix_1 = np.matrix([[734.5626,0,2847837],[0,797.2962,302.3755],[0,0,1]])#30.0783

camera_matrix_2 = np.matrix([[743.5223,0,333.2208],[0,802.5812,306.2985],[0,0,1]])#46.5610

dist_coeffs_1 = np.matrix([-0.1666,    0.1677,0,0,0])
dist_coeffs_2 = np.matrix([-0.2210,    0.8222,  0,0 ,0])

r = np.matrix([[0.9981,-0.0039,0.0615],[0.0034,1.00,0.0091],[-0.0616,-0.0088,0.9981]])
t = np.array([-62.2974,2.7549,4.8679])



r_1 = np.matrix([[0.9950,0.0896,0.0447],[-0.0819 ,   0.9852,   -0.1503],[-0.0575,    0.1459,    0.9876]])
p_1 = (1.0e+04)*np.matrix([[0.0653,    0.0128,    0.0000],[-0.0097,    0.1082,   -0.0000],[0.0254 ,   0.0680 ,   0.0001],[-8.4100 ,  -2.2334,   -0.0016 ]])

r_2 = np.matrix([[0.9921,    0.0928,    0.0849],[-0.0819,    0.9889,   -0.1243],[-0.0955,    0.1163,    0.9886]])
p_2 = (1.0e+05)*np.matrix([[0.0067,    0.0016,    0.0000],[ -0.0009,    0.0110,   -0.0000],[0.0024,    0.0069,    0.0000],[-2.0128,   -2.3393,   -0.0014]])

#r_1 = np.zeros((3,3),dtype='float')
#r_2 = np.zeros((3,3),dtype='float')
#p_1 = np.zeros((3,4),dtype='float')
#p_2 = np.zeros((3,4),dtype='float')

##########left image ############
'''new_cam_mat_1 , roi_1 = cv2.getOptimalNewCameraMatrix(camera_matrix_1,dist_coeffs_1,(cols,rows),0,(cols,rows))
img_1_rectified = cv2.undistort(img_1, camera_matrix_1, dist_coeffs_1, None, new_cam_mat_1)
# print ,camera_matrix_1
#x,y,w,h = roi_1
#img_1_rectified = img_1_rectified[y:y+h,x:x+h]

new_cam_mat_2 , roi_2 = cv2.getOptimalNewCameraMatrix(camera_matrix_2,dist_coeffs_2,(cols,rows),0,(cols,rows))
img_2_rectified = cv2.undistort(img_2, camera_matrix_2, dist_coeffs_2, None, new_cam_mat_2)
#x,y,w,h = roi_2
#img_2_rectified = img_2_rectified[y:y+h,x:x+h]
print roi_1,roi_2'''

( r_1,r_2,p_1,p_2,depth,roi_1,roi_2  ) = cv2.stereoRectify(camera_matrix_1,dist_coeffs_1,camera_matrix_2,dist_coeffs_2,(512,512),r,t,flags=0)


print p_1.shape	
print r_1,r_2
print p_1,p_2

newcameramtx_1, roi_1=cv2.getOptimalNewCameraMatrix(camera_matrix_1,dist_coeffs_1,(cols,rows),1,(cols,rows))

newcameramtx_2, roi_2=cv2.getOptimalNewCameraMatrix(camera_matrix_2,dist_coeffs_2,(cols,rows),1,(cols,rows))

undistortion_map_left,rectification_map_left   = cv2.initUndistortRectifyMap(camera_matrix_1,dist_coeffs_1,r_1,newcameramtx_1,(512,512),cv2.CV_32FC1)

undistortion_map_right,rectification_map_right   = cv2.initUndistortRectifyMap(camera_matrix_2,dist_coeffs_2,r_2,newcameramtx_2,(512,512),cv2.CV_32FC1)

print rectification_map_left.shape

normalised = np.sum(rectification_map_left)/(512.00*512.00)

img_1_rectified = cv2.remap(img_1,undistortion_map_left,rectification_map_left,cv2.INTER_NEAREST)
img_2_rectified = cv2.remap(img_2,undistortion_map_right,rectification_map_right,cv2.INTER_NEAREST)

color = (0,255,0)

'''cv2.line(img_1_rectified, (0,100), (511,100), color,1)
cv2.line(img_1_rectified, (0,256), (511,256), color,1)
cv2.line(img_2_rectified, (0,100), (511,100), color,1)
cv2.line(img_2_rectified, (0,256), (511,256), color,1)
cv2.line(img_1_rectified, (0,300), (511,300), color,1)
cv2.line(img_1_rectified, (0,400), (511,400), color,1)
cv2.line(img_2_rectified, (0,300), (511,300), color,1)
cv2.line(img_2_rectified, (0,400), (511,400), color,1)'''

cv2.imwrite('left_20.png',img_1_rectified)
cv2.imwrite('right_20.png',img_2_rectified)

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img_1_rectified,cmap='gray')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(img_2_rectified,cmap='gray')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(img_1,cmap='gray')
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(img_2,cmap='gray')

plt.show()  


'''cv2.imshow('left',img_1_rectified)
cv2.imshow('right',img_2)
cv2.waitKey(0)'''


