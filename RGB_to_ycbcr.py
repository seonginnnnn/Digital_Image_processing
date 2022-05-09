import numpy as np
import cv2

def cut():
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
image = cv2.imread("myface.jpg", cv2.IMREAD_COLOR)
print(np.shape(image))
Y = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
Cb = (image[:,:,2] - Y)*0.564 + 128
Cr = (image[:,:,0] - Y)*0.713 + 128

img_1 = np.array([Y,Cb,Cr],dtype=np.uint8)

R = Y + 1.403*(Cr-128)
G = Y - 0.714*(Cr-128) - 0.344*(Cb-128)
B = Y + 1.773*(Cb-128)

img_2 = np.dstack((R,G,B)).astype(np.uint8)

title1 = ['Y', 'Cr', 'Cb']
for i in range(len(img_1)):
    cv2.imshow("img_1[%d]-%s" %(i,title1[i]),img_1[i])
cut()
cv2.imshow('img_color',img_2)
cut()
