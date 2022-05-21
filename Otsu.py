import numpy as np
import cv2
import matplotlib.pyplot as plt

def Otsu_(img):
    img_f = img.flatten()
    img_h = np.bincount(img_f,minlength=256)
    num = np.arange(256)
    MN = np.sum(img_h)
    Pi = img_h/MN
    T = 0
    n = 0
    for i in range(1,255):
        k = i
        P1 = np.sum(Pi[1:k+1])
        P2 = 1 - P1
        if P2 == 0 or P1 == 0:
            continue
        m1 = np.sum(num[:k+1]*Pi[:k+1])/P1
        m2 = np.sum(num[k+1:]*Pi[k+1:])/P2
        mG = P1*m1 + P2*m2
        m = np.sum(num[:k+1]*Pi[:k+1])
        o2G = np.sum(((num - mG)**2)**Pi)
        o2B = ((mG*P1-m)**2)/(P1*(1-P1))
        n_ = o2B/o2G
        if n < n_:
            T = k
            n = n_
    img = np.where(img>T,255,0)
    print("T = ",T)
    return img
    
def Otsu(x,y,img):
    img[:x,:y] = Otsu_(img[:x,:y])
    img[:x,y:] = Otsu_(img[:x,y:])
    img[x:,:y] = Otsu_(img[x:,:y])
    img[x:,y:] = Otsu_(img[x:,y:])
    return img
    

img1 = cv2.imread("9_5_4.jpg",cv2.IMREAD_GRAYSCALE)
img1_Ots = Otsu(100,100,img1)
cv2.imshow('9_5_4',img1_Ots)
img2 = cv2.imread("15_5_2.jpg",cv2.IMREAD_GRAYSCALE)
img2_Ots = Otsu(135,135,img2)
cv2.imshow('15_5_2',img2_Ots)
img3 = cv2.imread("17_5_7.jpg",cv2.IMREAD_GRAYSCALE)
img3_Ots = Otsu(135,100,img3)
cv2.imshow('17_5_7',img3_Ots)
