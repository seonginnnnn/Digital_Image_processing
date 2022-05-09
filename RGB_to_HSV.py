import numpy as np
import cv2
import matplotlib.pyplot as plt
image = cv2.imread("abc.jpg", cv2.IMREAD_COLOR)

R,G,B = image[:,:,2]/255,image[:,:,1]/255,image[:,:,0]/255
V = np.max(image,axis=2)/255
VM = np.min(image,axis=2)/255

D = V - VM
S = (np.where(V != 0,D/V,0)*255).astype(np.uint8)
H = np.where(V==0,0,np.where(V==R,60*(G - B)/D,
             np.where(V==G,120+(60*(B - R)/D),
                      240+(60*(R-G)/D))))
H = (np.where(H<0,H + 360,H)/2).astype(np.uint8)

cv2.imshow('H',H)
cv2.imshow('S',S)
cv2.imshow('V',V)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_2 = np.dstack((H,S,V)).astype(np.uint8)
histo = np.zeros((np.max(H)+1,np.max(S)+1),np.uint8)
for i in range(len(H)):
    for j in range(len(H[0])):
        histo[H[i,j],S[i,j]] += 1

a = 20
histo = ((histo/a) * 255).astype(np.uint8)
plt.pcolor(histo)
plt.xlabel('Saturation')
plt.ylabel('Hue')
plt.colorbar()
plt.show()
