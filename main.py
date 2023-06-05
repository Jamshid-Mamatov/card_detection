import cv2
import numpy as np

# first duty is to extract key points of card like number and cue
def thresh_img(path):
    img=cv2.imread(path)

    # cv2.imshow('test',img)

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    
    thresh=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    return thresh

path="test_img/document.jpeg"
thresh=thresh_img(path=path)

contours, hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# print(hierarchy)
img=cv2.imread(path)

# res=cv2.drawContours(img,contours,-1,255,2)



index_sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)

for i in index_sort:
    per=cv2.arcLength(contours[i],True)
    curve=cv2.approxPolyDP(contours[i],0.01*per,True)
  
    if len(curve)==4:
        x1,y1=curve[0][0]
        x2,y2=curve[0][-1]
        
        cv2.rectangle(img,[x1,y1],[x2,y2],(255,0,0),2)

for c in contours:
  x= cv2.minAreaRect(c)
 
    # Make sure contour area is large enough
  if (cv2.contourArea(c)) > 30000 and (cv2.contourArea(c)) < 100000:
    box = cv2.boxPoints(x)
    box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
    cv2.drawContours(img, [box], 0, 255)
cv2.imshow("res",img)
cv2.imshow("thresh",thresh)

cv2.waitKey(0)

cv2.destroyAllWindows()