import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

def four_point_transform(image, pts):
    """
    https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    """
    # obtain a consistent order of the points and unpack them
    # individually
    # rect = order_points(pts)
    rect = np.transpose(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    print(dst)
    print(maxHeight, maxWidth)

    M = cv2.getPerspectiveTransform(np.float32(rect), np.float32(dst))
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    """
    https://math.stackexchange.com/questions/2796055/3d-coordinate-rotation-using-roll-pitch-yaw
    """
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                               
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

# Intrinsics (calib module)
K = np.array([[676.74,0,317.4],[0,863.29,252.459],[0,0,1]])

# Extrinsics (mounting parameters)
angles = np.array([90,90,0])*np.pi/180
R = eulerAnglesToRotationMatrix(angles) # Rotation matrix  WCS->CCS
T = np.array([[0.0],[-0.156],[0.0]]) # WCS expressed in camera coordinate system 
Rt = np.hstack((R,T))

xworld = 0.6
yworld = -0.16
lengthx = 1.0
lengthy = 2*-1*yworld
# TODO: Define this according to lengthx and lengthy
Xw = np.array([[xworld + lengthx, xworld + lengthx, xworld, xworld          ], \
               [yworld + lengthy, yworld          , yworld, yworld + lengthy], \
               [0.0             , 0.0             , 0.0   , 0.0             ],\
               [1.0             , 1.0             , 1.0   , 1.0             ]])

print(Xw)
img_path = "C:/Users/user/Ponc/terrinus/train/28.jpeg"
img = cv2.imread(img_path)
print(img.shape)
xp = np.zeros((2,4))
for i in range(4):
    pcc = np.matmul(Rt,Xw[:,i])
    ximg = np.matmul(K, pcc)
    ximg = ximg[:2]/ximg[2]
    cv2.circle(img,(int(ximg[0]), int(ximg[1])), 5,(0,0,255))
    print("Point in image = (", int(ximg[0]), ", ", int(ximg[1]), ")")
    xp[:,i] = np.array([int(ximg[0]),int(ximg[1])])
cv2.imshow('a', img)
cv2.waitKey(0)


BEV_img = four_point_transform(img, xp)
cv2.imshow('BEV', BEV_img)
cv2.waitKey(0)
