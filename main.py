import cv2
import numpy as np
import random as rng
# first duty is to extract key points of card like number and cue
def thresh_img(path):
    img=cv2.imread(path)
    img=cv2.resize(img,(640,640))
    

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),0)
    blur = cv2.blur(blur,(5,5))
    # cv2.imshow("blur",blur)
    # thresh=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    # ret,thresh=cv2.threshold(blur,138,255,cv2.THRESH_BINARY)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([0,0,255])
    upper_blue = np.array([255,255,255])
    thresh = cv2.inRange(hsv, lower_blue, upper_blue)
    return thresh


# def draw_contour(path):
#     thresh=thresh_img(path=path)
#     # cv2.imshow("thresh",thresh)
#     img=cv2.imread(path)
#     img=cv2.resize(img,(640,640))
#     contours, hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
#     index_sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)
#     # cv2.s(img,contours,-1,(255,0,0))
    
#     for i in index_sort:
        
        
#         contourind=hierarchy[0][i][3]
#         per=cv2.arcLength(contours[i],True)
#         curve=cv2.approxPolyDP(contours[i],0.01*per,True)
#         # print(len(curve))
        
#         if contourind!=-1 and len(curve)==4:
#             # print(curve)
#             # img=cv2.drawContours(img,contours,i,(0,0,255),2)
#             x1,y1=curve[0][0]
#             x2,y2=curve[1][0]
#             x3,y3=curve[2][0]
#             x4,y4=curve[3][0]
#             cv2.circle(img,(x1,y1),5,(0,0,255),-1)
#             cv2.circle(img,(x2,y2),5,(0,0,255),-1)
#             cv2.circle(img,(x3,y3),5,(0,0,255),-1)
#             cv2.circle(img,(x4,y4),5,(0,0,255),-1)
#     return img

path="test_img/card.jpg"

# img=draw_contour(path)

# res=cv2.drawContours(img,contours,-1,255,2)

def drawboundingbox(path):

    img=cv2.imread(path)
    img=cv2.resize(img,(640,640))
    thresh=thresh_img(path=path)
    cv2.imshow("thresh",thresh)
    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        per=cv2.arcLength(contours[i],True)
        contours_poly[i] = cv2.approxPolyDP(c, 0.01*per, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        area=cv2.contourArea(contours[i])
        # print(area)
        # cv2.drawContours(drawing, contours_poly, i, color)
        if area>50000:
            cv2.rectangle(img, (int(boundRect[i][0]), int(boundRect[i][1])), \
            (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
  
    
    return img

draw=drawboundingbox(path)

cv2.imshow("draw",draw)
# parent_child_dict={}
# for idx,value in enumerate(hierarchy[0]):
    
#     if value[3]!=-1 and value[3] not in parent_child_dict:
#       parent_child_dict[value[3]]=[]
#     if value[3]!=-1:
#       parent_child_dict[value[3]].append(idx)
# print(parent_child_dict)
# for c in contours:
# #   x= cv2.minAreaRect(c)
 
# #     # Make sure contour area is large enough
# #   if (cv2.contourArea(c)) > 30000 and (cv2.contourArea(c)) < 100000:
# #     box = cv2.boxPoints(x)
# #     box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
# #     cv2.drawContours(img, [box], 0, 255)

#     pass
# cv2.imshow("res",img)
# cv2.imshow("thresh",thresh)

cv2.waitKey(0)

cv2.destroyAllWindows()