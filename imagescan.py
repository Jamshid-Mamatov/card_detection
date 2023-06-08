import cv2
import numpy as np
import random as rng


path="test_img/4.jpg"
img=cv2.imread(path)
img=cv2.resize(img,(640,640))
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel = np.ones((5,5),np.uint8)

gradient=cv2.morphologyEx(gray,cv2.MORPH_GRADIENT,kernel)

# print(gradient.shape)

# contours,hierarchy=cv2.findContours(gradient,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE )

# raw_img=cv2.drawContours(img,contours,-1,(255,0,0),2)



gradient=np.where(gradient>10,gradient,0)
gradient = cv2.morphologyEx(gradient, cv2.MORPH_OPEN, kernel)
thresh = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)



contours,hierarchy=cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE )


contours_poly = [None]*len(contours)
boundRect = [None]*len(contours)

for i, c in enumerate(contours):
    per=cv2.arcLength(contours[i],True)
    contours_poly[i] = cv2.approxPolyDP(c, 0.01*per, True)
    # boundRect[i] = cv2.boundingRect(contours_poly[i])
drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    area=cv2.contourArea(contours[i])
    per=cv2.arcLength(contours[i],True)
    curve=cv2.approxPolyDP(contours[i],0.01*per,True)
    # print(area)
    # cv2.drawContours(drawing, contours_poly, i, color)
    if area>5000 and len(curve)==4:
        pass
        print(contours_poly[i])
        x1,y1=curve[0][0]
        x2,y2=curve[1][0]
        x3,y3=curve[2][0]
        x4,y4=curve[3][0]
        cv2.circle(img,(x1,y1),5,(0,0,255),-1)
        cv2.circle(img,(x2,y2),5,(0,0,255),-1)
        cv2.circle(img,(x3,y3),5,(0,0,255),-1)
        cv2.circle(img,(x4,y4),5,(0,0,255),-1)
        # cv2.rectangle(img, (int(boundRect[i][0]), int(boundRect[i][1])), \
        # (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
# raw_img=cv2.drawContours(img,contours,-1,(255,0,0),2)


# print(len(contours))
cv2.imshow('gradient',gradient)
cv2.imshow('drawed',img)

cv2.waitKey(0)

cv2.destroyAllWindows()