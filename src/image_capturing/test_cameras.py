import numpy as np
import cv2

# cv2.VideoCapture.set(CV_CAP_PROP_FRAME_WIDTH = 512,CV_CAP_PROP_FRAME_HEIGHT =512)

cap_left = cv2.VideoCapture(2)
cap_right = cv2.VideoCapture(1)

ret = cap_right.set(3,512)
ret = cap_right.set(4,512)
ret = cap_left.set(3,512)
ret = cap_left.set(4,512)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)


while(True):
    # Capture frame-by-frame
    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()
    #frame_l = cv2.flip(frame_l,0)
    #frame_l = cv2.flip(frame_l,1)
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #left = cv2.cvtColor(frame_l,cv2.COLOR_BGR2GRAY)
    #right = cv2.cvtColor(frame_r,cv2.COLOR_BGR2GRAY)
    #disparity = stereo.compute(left,right)
    # Display the resulting frame
    cv2.imshow('frame_l',frame_l)
    cv2.imshow('frame_r',frame_r)
    #cv2.imshow('disparity',disparity)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()