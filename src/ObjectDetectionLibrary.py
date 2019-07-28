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

def find_object_rect(IMG_left, IMG_right):
	img = IMG_right
	mask = np.zeros(img.shape[:2],np.uint8)
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	rect = (25,25,450,400)
	#t1 = time.time()
	cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
	#print(time.time() - t1)
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	img = img*mask2[:,:,np.newaxis]
	return img
	
	
def find_object_mask(IMG_left, IMG_right,disparity):	
	img = IMG_left
	min_disp = disparity.min()
	max_disp = disparity.max()
	threshold = float(max_disp - min_disp)/float(100)
	x = max_disp - 45*threshold
	y = float(max_disp - min_disp)/float(2)
	mask = np.zeros(img.shape[:2],np.uint8)
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	newmask = np.zeros(disparity.shape, dtype = int)
	
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if (disparity[i][j] <= 20*threshold):
				newmask[i][j] = -1			#(sure background)
			elif (x <= disparity[i][j] <= max_disp):
				newmask[i][j] = 1
	# where ever it is marked white (sure foreground), change mask=1
	# where ever it is marked black (sure background), change mask=0
	mask[newmask == -1] = 0
	mask[newmask == 0] = 1
	rect = (50,50,450,200)
	mask, bgdModel, fgdModel = cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
	mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	img = img*mask[:,:,np.newaxis]
	
	return img
	
			
	
