import cv2
import numpy as np

# first duty is to extract key points of card like number and cue

img=cv2.imread("test_img/card.jpg")

# cv2.imshow('test',img)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
img_w,img_h=img.shape[:2]

thresh=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)


cv2.imshow("thresh",thresh)
cv2.waitKey(0)

cv2.destroyAllWindows()