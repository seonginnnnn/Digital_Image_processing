import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('myface.jpg')
R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B

#모서리 부분 좌표 찾기
ang = (np.pi/180)*54
a = np.array([[0,599,599,0],[0,0,799,799]])
h0 = np.array([[np.cos(ang),-1*np.sin(ang)],[np.sin(ang),np.cos(ang)]])
edge = h0.dot(a)
a_x = np.min(edge[0])
a_y = np.min(edge[1])
b_x = edge[0] - a_x
b_y = edge[1] - a_y
c_x = np.max(b_x)
c_y = np.max(b_y)
d_x = (int(a_x))
d_y = (int(a_y))
temp = np.zeros((int(c_y)+1,int(c_x)+1))
h = np.array([[np.cos(ang),-1*np.sin(ang),-1*d_x],
              [np.sin(ang),np.cos(ang),-1*d_y],
              [0,0,1]])
#동차
for i in range(800):
    for j in range(600):
        xy0 = np.array([[j],[i],[1]])
        xy1 = h.dot(xy0)
        temp[int(xy1[1]),int(xy1[0])] = imgGray[i,j]

#좌표찾기
points_x = np.arange(0,len(temp[0]))
points_y = np.arange(0,len(temp))
ys, xs = np.meshgrid(points_y, points_x)
h_ = np.linalg.inv(h)
z_x,z_y = np.where((0<(h_[0][0]*(xs+d_x)+h_[0][1]*(ys+d_y)))&
             ((h_[0][0]*(xs+d_x)+h_[0][1]*(ys+d_y))<598)&
             (0<(h_[1][0]*(xs+d_x)+h_[1][1]*(ys+d_y)))&
             ((h_[1][0]*(xs+d_x)+h_[1][1]*(ys+d_y))<798)&
                   (temp[ys,xs]==0))

plt.imshow(temp,'gray')
plt.show()
temp2 = temp
#후진 사상 보간
for i in range(len(z_x)):
    b1 = np.array([[z_x[i]],[z_y[i]],[1]])
    b2 = h_.dot(b1)
    temp[z_y[i],z_x[i]] = imgGray[int(b2[1]),int(b2[0])]

#1차 보간
def find_x_noZero(temp2_0,cnt_0,RL):
    if temp2_0[z_y[i],z_x[i]+cnt_0] == 0:
        cnt_0 = find_x_noZero(temp2_0,cnt_0+RL,RL)
    return cnt_0

def find_y_noZero(temp2_1,cnt_1,UD):
    if temp2_1[z_y[i]+cnt_1,z_x[i]] == 0:
        cnt_1 = find_y_noZero(temp2_1,cnt_1+UD,UD)
    return cnt_1

for i in range(len(z_x)):
    cnt_xr = find_x_noZero(temp2,1,1)
    cnt_xl = find_x_noZero(temp2,-1,-1)
    cnt_yu = find_x_noZero(temp2,1,1)
    cnt_yd = find_x_noZero(temp2,-1,-1)
    
    m_x = ((temp2[z_y[i],z_x[i]+cnt_xr]-temp2[z_y[i],z_x[i]+cnt_xl])/(cnt_xr-cnt_xl))
    n_x = temp2[z_y[i],z_x[i]+cnt_xl] - (m_x*(z_x[i]+cnt_xl))
    temp_x = m_x*z_x[i]+n_x
    
    m_y = ((temp2[z_y[i]+cnt_yu,z_x[i]]-temp2[z_y[i]+cnt_yd,z_x[i]])/(cnt_yu-cnt_yd))
    n_y = temp2[z_y[i]+cnt_yd,z_x[i]] - (m_y*(z_y[i]+cnt_yd))
    temp_y = m_y*z_y[i]+n_y
    
    ans = (temp_x + temp_y)/2
    temp2[z_y[i],z_x[i]] = ans

plt.subplot(1,2,1)
plt.imshow(temp,'gray')
plt.subplot(1,2,2)
plt.imshow(temp2,'gray')
plt.show()
