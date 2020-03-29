import cv2
import numpy as np
from numpy.linalg import inv
img_path = r"C:\Users\user\Ponc\terrinus\annotation\train\17.jpeg"
img = cv2.imread(img_path)
width, height, c = img.shape
# Rc represents a -90 rotation wrt x axis (top view)
Rc = np.array([ [1.0,  0.0, 0.0],\
                [0.0,  0.0, 1.0],\
                [0.0, -1.0, 0.0] \
                ])
ang_d = -90.0
ang_r = ang_d * np.pi / 180.0
Rc = np.array([ [1.0,  0.0, 0.0],\
                [0.0,  np.cos(ang_r), -np.sin(ang_r)],\
                [0.0, np.sin(ang_r), np.cos(ang_r)] \
                ])
# Rc = np.array([ [np.cos(ang_r),  0.0, np.sin(ang_r)],\
#                 [0.0,  1.0, 0.0],\
#                 [-np.sin(ang_r), 0.0, np.cos(ang_r)] \
#                 ])

# Rc = np.array([ [np.cos(ang_r), -np.sin(ang_r), 0.0],\
#                 [np.sin(ang_r), np.cos(ang_r), 0.0],\
#                 [0.0,  0.0, 1.0]])

tc = np.array([0.0, 0.0, 0.0])
nc = np.array([0.0, 0.0, 1.0])
dc = 0.15

K = np.array([[646.74, 0.0, 317.4],\
              [0, 863.29, 252.549],\
              [0.0, 0.0, 1.0]])

Hc = np.linalg.inv(Rc) + np.matmul(-tc, nc)/dc
Hc = np.matmul(np.matmul(K,inv(Rc)), inv(K))
Hc_img = np.matmul(K, Hc)

top_view = cv2.warpPerspective(img[300:,:], Hc,(width,height))
cv2.imshow('img', img)
cv2.imshow('top_view', top_view)
cv2.waitKey(0)