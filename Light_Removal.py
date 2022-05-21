import numpy as np
import cv2
import matplotlib.pyplot as plt

def Otsu(img):
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
    return img

def img_cut(img,A,B,x1,x2,y1,y2):
    x = np.linspace(x1,x2,x2-x1+1)
    y = np.linspace(y1,y2,y2-y1+1)
    X,Y = np.meshgrid(x,y)
    X = X.astype(int).reshape((-1,1))
    Y = Y.astype(int).reshape((-1,1))
    Z = np.ones_like(X)
    i = img[X,Y].reshape((-1,1))
    j = np.hstack((X,Y,Z))
    A = np.vstack((A,j))
    B = np.vstack((B,i))
    
    return A,B

def cut_light(S,img):
    A = np.array([-1,-1,-1])
    B = np.array([-1])
    A,B = img_cut(img,A,B,0,S,0,S)
    A,B = img_cut(img,A,B,0,S,len(img)-S-1,len(img)-1)
    A,B = img_cut(img,A,B,len(img[0])-S-1,len(img[0])-1,0,S)
    A,B = img_cut(img,A,B,len(img[0])-S-1,len(img[0])-1,len(img)-S-1,len(img)-1)
    A = A[1:]
    B = B[1:]
    abc = np.linalg.inv((A.T.dot(A))).dot(A.T).dot(B)
    x = np.linspace(0,len(img)-1,len(img))
    y = np.linspace(0,len(img[0])-1,len(img[0]))
    Y,X = np.meshgrid(x,y)
    X = X.astype(int).reshape((-1,1))
    Y = Y.astype(int).reshape((-1,1))
    one_s = np.ones_like(X)
    A_ = np.hstack((X,Y,one_s))
    b_ = A_.dot(abc).reshape((200,200))
    return b_

def print_(img,size):
    img_L = cut_light(size,img)
    cv2.imshow('0',img)
    cv2.imshow('1',img_L.astype(np.uint8))
    img_cut = img - img_L
    img_cut = img_cut - np.min(img_cut)
    img_cut = img_cut.astype(np.uint8)
    img_Ots = Otsu(img_cut).astype(np.uint8)
    
    cv2.imshow('2',img_cut)
    cv2.imshow('3',img_Ots)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
img1 = cv2.imread("9_5_4.jpg",cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("15_5_2.jpg",cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("17_5_7.jpg",cv2.IMREAD_GRAYSCALE)
print_(img1,50)
print_(img2,50)
print_(img3,50)
