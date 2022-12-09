import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img0 = mpimg.imread('face_0.jpg')
R0, G0, B0 = img0[:,:,0], img0[:,:,1], img0[:,:,2]
imgGray = 0.2989 * R0 + 0.5870 * G0 + 0.1140 * B0
plt.imshow(imgGray,'gray')
x1_a = plt.ginput(20)
x1_b = np.array(x1_a)
x1_c = np.around(x1_b)
x1 = x1_c.astype(np.int64)

img1 = mpimg.imread('face_1.jpg')
R1, G1, B1 = img1[:,:,0], img1[:,:,1], img1[:,:,2]
imgGray1 = 0.2989 * R1 + 0.5870 * G1 + 0.1140 * B1
plt.imshow(imgGray1,'gray')
x2_a = plt.ginput(20)
x2_b = np.array(x2_a)
x2_c = np.around(x2_b)
x2 = x2_c.astype(np.int64)

x1 = np.array([[ 68, 157],
       [210, 157],
       [358, 157],
       [488, 157],
       [ 68, 280],
       [210, 280],
       [358, 280],
       [488, 280],
       [ 68, 403],
       [210, 403],
       [358, 403],
       [488, 403],
       [ 68, 526],
       [210, 526],
       [358, 526],
       [488, 526],
       [ 68, 667],
       [210, 667],
       [358, 667],
       [488, 667]])

##x2 = np.array([[ 77, 179],
##       [215, 174],
##       [359, 174],
##       [482, 172],
##       [ 77, 298],
##       [217, 298],
##       [359, 294],
##       [486, 291],
##       [ 79, 420],
##       [215, 420],
##       [359, 418],
##       [481, 420],
##       [ 73, 541],
##       [216, 545],
##       [359, 538],
##       [485, 544],
##       [ 79, 676],
##       [213, 667],
##       [358, 684],
##       [479, 689]])  

temp = np.zeros((800,600))

for i in range(4):
    for j in range(3):
##변화값 구하기
        xy_ = np.array([[x1[i*4+j][0],x1[i*4+j+1][0],x1[(i+1)*4+j+1][0],x1[(i+1)*4+j][0]],
                      [x1[i*4+j][1],x1[i*4+j+1][1],x1[(i+1)*4+j+1][1],x1[(i+1)*4+j][1]]])
        xy_0 = np.array([[x2[i*4+j][0],x2[i*4+j+1][0],x2[(i+1)*4+j+1][0],x2[(i+1)*4+j][0]],
                       [x2[i*4+j][1],x2[i*4+j+1][1],x2[(i+1)*4+j+1][1],x2[(i+1)*4+j][1]]])

        h = np.array([xy_[0]*xy_[1],xy_[0],xy_[1],[1,1,1,1]])
        h_li = np.linalg.inv(h)
        ang = xy_0.dot(h_li)
        abcd = ang.dot(h)
        a = np.min(xy_[0])
        b = np.max(xy_[0])
        c = np.min(xy_[1])
        d = np.max(xy_[1])
       
        for k in range(a,b):
            for l in range(c,d):
                thr = np.array([[k*l],[k],[l],[1]])
                end_xy = ang.dot(thr)
                temp[int(end_xy[1]),int(end_xy[0])] = imgGray[l,k]
                
## 보간법
## 반대로
for i in range(4):
    for j in range(3):
        xy_0 = np.array([[x1[i*4+j][0],x1[i*4+j+1][0],x1[(i+1)*4+j+1][0],x1[(i+1)*4+j][0]],
                      [x1[i*4+j][1],x1[i*4+j+1][1],x1[(i+1)*4+j+1][1],x1[(i+1)*4+j][1]]])
        xy_ = np.array([[x2[i*4+j][0],x2[i*4+j+1][0],x2[(i+1)*4+j+1][0],x2[(i+1)*4+j][0]],
                       [x2[i*4+j][1],x2[i*4+j+1][1],x2[(i+1)*4+j+1][1],x2[(i+1)*4+j][1]]])

        h = np.array([xy_[0]*xy_[1],xy_[0],xy_[1],[1,1,1,1]])
        h_li = np.linalg.inv(h)
        ang = xy_0.dot(h_li)
        abcd = ang.dot(h)
        a = np.min(xy_[0])
        b = np.max(xy_[0])
        c = np.min(xy_[1])
        d = np.max(xy_[1])
        _a = np.min(xy_0[0])
        _b = np.max(xy_0[0])
        _c = np.min(xy_0[1])
        _d = np.max(xy_0[1])
        for x in range(a,b):
            for y in range(c,d):
                thr = np.array([[x*y],[x],[y],[1]])
                end_xy = ang.dot(thr)
                if end_xy[0] >= _a and end_xy[0] <= _b and end_xy[1] >= _c and end_xy[1] <= _d:
                    if temp[y,x] == 0:
                        temp[y,x] = imgGray[int(end_xy[1]),int(end_xy[0])]

        

plt.imshow(temp,'gray')
plt.show()
