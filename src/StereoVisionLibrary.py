##################################################
# Filename: StereoVisionLibrary.py
# Function:
#	1. rectifies the right and left image
#	2. computes disparity using SGBM
#	3. creates point cloud in ply file
#	4. computes distance/depth from disparity map
##################################################

import cv2
import numpy as np
import PointCloudLibrary as pcl 

# Camera Parameters
camera_matrix_1 = np.matrix([[734.5626,0,284.7837],[0,797.2962,302.3755],[0,0,1]])#30.0783
camera_matrix_2 = np.matrix([[743.5223,0,333.2208],[0,802.5812,306.2985],[0,0,1]])#46.5610
dist_coeffs_1 = np.matrix([-0.1666,    0.1677,0,0,0])
dist_coeffs_2 = np.matrix([-0.2210,    0.8222,  0,0 ,0])
r = np.matrix([[0.9981,-0.0039,0.0615],[0.0034,1.00,0.0091],[-0.0616,-0.0088,0.9981]])
t = np.array([-62.2974,2.7549,4.8679])

def compute_disparity(img_l,img_r):

	rows,cols = img_l.shape[:2]
	color = (0,255,0)
	
	# rectification to find Q matrix
	Q = rectify(img_l,img_r)
	
	# Semi-global-block-matching
	stereo = SGBM()
	
	# computing disparity
	disparity = stereo.compute(img_l, img_r)

	# creating point cloud in ply form
	Image3D = cv2.reprojectImageTo3D(disparity, Q, ddepth=cv2.CV_32F)
	colors = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
	mask = disparity > disparity.min()
	out_points = Image3D[mask]
	out_colors = colors[mask]
	out_fn = 'out.ply'
	pcl.write_ply('out.ply', out_points, out_colors)
	print('%s saved' % 'out.ply')
	
	return disparity
	
def compute_depth(obj, disparity):
	f1 = (camera_matrix_1[0,0] + camera_matrix_1[1,1])/2.0
	f2 = (camera_matrix_2[0,0] + camera_matrix_2[1,1])/2.0
	f = (f1 + f2)/2.0
	T = 5.0
	kernel = np.ones((10,10),np.uint8)
	obj_color = obj
	obj = cv2.cvtColor(obj,cv2.COLOR_BGR2GRAY)
	ret,thresh1 = cv2.threshold(obj,30,255,cv2.THRESH_BINARY)
	print(obj.shape)
	erosion = cv2.erode(thresh1,kernel,iterations = 2)
	print(erosion.dtype)
	dilation = cv2.dilate(erosion,kernel,iterations = 2)
	print(dilation.dtype)
	im2,contours,hierarchy = cv2.findContours(dilation, 1, 2)
	max_cnt = 0
	max_area= 0
	for cnt in contours:
	    area = cv2.contourArea(cnt)
	    if(area > max_area):
	    	max_cnt = cnt
	    	max_area= area

	x,y,w,h = cv2.boundingRect(max_cnt)
	area = cv2.contourArea(max_cnt)
	        
	depth  = np.zeros(disparity.shape, dtype = float)
	final_depth = 0
	final_disparity=0
	count=0
	for i in range(x,x+w):
		for j in range(y,y+h):
			if(obj[i,j] != 0):
				final_disparity += disparity[i,j]
				depth[i,j] = abs(float(f*T)/float(disparity[i,j]))
				#print(depth[i,j])
				final_depth += depth[i,j]
			else:
				count +=1
	area= w*h
	#print(area)
	final_depth = final_depth/(area-count)
	cv2.rectangle(obj_color,(x,y),(x+w,y+h),(0,255,0),2)
	ellipse= cv2.fitEllipse(max_cnt)
	#cv2.ellipse(obj_color,ellipse,(0,255,0),2)
	#cv2.imshow('obj',obj_color)
	if(disparity[x+w/2,y+h/2] >0):
		depth_center = T*f/(abs(disparity[x+w/2,y+h/2])+32000 )
	else:
		depth_center = T*f/(32000 -abs(disparity[x+w/2,y+h/2]) )
	
	print (final_disparity)
	final_disparity = final_disparity/(area - count)
	
	print(final_disparity)
	
	print (area-count)
	print (final_depth)
	print( T*f/final_disparity )
	print (depth_center)
	#print(disparity)
	return final_depth*10
		
def rectify(img_l,img_r):
	r_1,r_2,p_1,p_2,q,roi_1,roi_2 = cv2.stereoRectify(camera_matrix_1,dist_coeffs_1,camera_matrix_2,dist_coeffs_2,(512,512),r,t,flags=0)
	return q

def SGBM():	
	window_size = 11
	min_disp = -40
	num_disp = 12*16
	uniquenessRatio = 1
	speckleWindowSize = 50
	speckleRange = 1
	disp12MaxDiff = 10
	P1 = 10*3*window_size**2
	P2 = 32*3*window_size**2	
	stereo = cv2.StereoSGBM_create(minDisparity = min_disp, numDisparities = num_disp,blockSize = window_size,uniquenessRatio = uniquenessRatio,speckleWindowSize = speckleWindowSize,speckleRange=speckleRange,disp12MaxDiff = disp12MaxDiff,P1 = P1,P2 = P2)
	
	return stereo
