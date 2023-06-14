import cv2
import numpy as np
import random as rng



path="test_img/messi5.jpg"

def mask(path):

    img=cv2.imread(path)
    # img=cv2.resize(img,(640,640))
    # hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask = np.zeros(img.shape[:2],np.uint8)
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (50,50,450,290)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    cv2.rectangle(img,(50,50),(50+450,50+290),(255,0,0),2)
    
    # lower_ = np.array([0,0,130])
    # upper_ = np.array([15,40,255])
    # thresh=cv2.inRange(hsv,lower_,upper_)
    cv2.imshow("thresh",img)
    return img

mask(path)

cv2.waitKey(0)

cv2.destroyAllWindows()