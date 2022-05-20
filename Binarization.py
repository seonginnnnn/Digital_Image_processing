import numpy as np
import cv2
import matplotlib.pyplot as plt

def jinfire(img):
    imgGray = np.array((img[:,:,2]))
    imgGray_f = imgGray.flatten()
    imgGray_h = np.bincount(imgGray_f,minlength=256)
    plt.plot(imgGray_h)
    plt.show()
    num = np.arange(256)
    T = 120
    T_o = 120
    while True:
        G1 = imgGray_h[:T]
        G1_ran = num[:T]
        G2 = imgGray_h[T:]
        G2_ran = num[T:]
        m1 = np.sum(G1*G1_ran)/np.sum(G1)
        m2 = np.sum(G2*G2_ran)/np.sum(G2)
        T_new = int((m1+m2)/2)
        if T == T_new:
            break
        elif np.abs(T - T_new) <=  (imgGray_h[T]-imgGray_h[T_new])/(T-T_new):
            T = T_new
            break
        else:
            T = T_new
            
    img_2jf = np.where(imgGray >= T,255,0).astype(np.uint8)
    print("T = ",T)
    return img_2jf

img1 = cv2.imread("9_5_4.jpg",cv2.IMREAD_COLOR)
img2 = cv2.imread("15_5_2.jpg",cv2.IMREAD_COLOR)
img3 = cv2.imread("17_5_7.jpg",cv2.IMREAD_COLOR)
cv2.imshow('abc1',jinfire(img1))
cv2.imshow('abc2',jinfire(img2))
cv2.imshow('abc3',jinfire(img3))
