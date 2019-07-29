import numpy as np
import cv2
import os

# cv2.VideoCapture.set(CV_CAP_PROP_FRAME_WIDTH = 512,CV_CAP_PROP_FRAME_HEIGHT =512)

cap_left = cv2.VideoCapture(0)
# cap_right = cv2.VideoCapture(1)

# ret = cap_right.set(3,512)
# ret = cap_right.set(4,512)
ret = cap_left.set(3,512)
ret = cap_left.set(4,512)

# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

count_left= 27
count_right = 27

def on_click_right(cap):
    print "right clicked"
    ret,image = cap.read()
    global count_left
    side = 'right'
    # for side, image in zip(("left", "right"), images):
    filename = "{}_{}.png".format(side, count_right)
    output_path = os.path.join('image/', filename)
    cv2.imwrite(output_path, image)
    count_left +=1
    pass

def on_click_left(cap):
    print "left clicked"
    ret,image = cap.read()
    global count_right
    side = 'left'
    # for side, image in zip(("left", "right"), images):
    filename = "{}_{}.png".format(side, count_left)
    output_path = os.path.join('image/', filename)
    cv2.imwrite(output_path, image)
    count_right +=1
    pass


while(True):
    # Capture frame-by-frame
    ret_l, frame_l = cap_left.read()
    # ret_r, frame_r = cap_right.read()
    #frame_l = cv2.flip(frame_l,0)
    #frame_l = cv2.flip(frame_l,1)
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #left = cv2.cvtColor(frame_l,cv2.COLOR_BGR2GRAY)
    #right = cv2.cvtColor(frame_r,cv2.COLOR_BGR2GRAY)
    #disparity = stereo.compute(left,right)
    # Display the resulting frame
    cv2.imshow('frame_l',frame_l)
    # cv2.imshow('frame_r',frame_r)
    #cv2.imshow('disparity',disparity)
    if cv2.waitKey(1) & 0xFF == ord('r'):
                on_click_right(cap_left)
    if cv2.waitKey(1) & 0xFF == ord('l'):
                on_click_left(cap_left)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()