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
    return warped,M

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
yworld = +0.25
lengthx = 1.0
lengthy = 2*yworld
Xw = np.array([[xworld + lengthx, xworld + lengthx, xworld,           xworld],\
               [yworld,           yworld - lengthy, yworld - lengthy, yworld],\
               [0.0             , 0.0             , 0.0   , 0.0             ],\
               [1.0             , 1.0             , 1.0   , 1.0             ]])


img_path = "C:/Users/user/Ponc/terrinus/train/01.jpeg"
img = cv2.imread(img_path)
xp = np.zeros((2,4))
colors = [(255,255,255),(255,0,0),(0,255,0),(0,0,255)]

num_x = 5
num_y = 5
grid_points = np.zeros((4,num_x*num_y))
grid_points[3,:] = 1.0
for a in range(num_x * num_y):
    i = a//num_y
    j = a%num_y
    x = xworld + lengthx - ((i/(num_x-1))*lengthx)
    y = yworld - ((j/(num_y-1))*lengthy)
    grid_points[0, a] = x
    grid_points[1, a] = y

points_grid_cam_coord = np.matmul(Rt,grid_points)
points_grid_img = np.matmul(K,points_grid_cam_coord)
points_grid_img = points_grid_img[:2,:]/points_grid_img[2]
points_grid_img = np.round(points_grid_img)
print(points_grid_img.shape)
print("fransecet")
for i in range(num_x*num_y):
    pccg = np.matmul(Rt,grid_points[:,i])
    ximg_g = np.matmul(K,pccg)
    ximg_g = ximg_g[:2]/ximg_g[2]
    cv2.circle(img,(int(ximg_g[0]), int(ximg_g[1])), 5, (0,255,0))

for i in range(4):
    pcc = np.matmul(Rt,Xw[:,i])
    ximg = np.matmul(K, pcc)
    ximg = ximg[:2]/ximg[2]
    cv2.circle(img,(int(ximg[0]), int(ximg[1])), 5, (0,255,0))
    # print("Point in image = (", int(ximg[0]), ", ", int(ximg[1]), ")")
    xp[:,i] = np.array([int(ximg[0]),int(ximg[1])])


# cv2.imshow('a', img)
# cv2.waitKey(0)

BEV_img,M = four_point_transform(img, xp)
# cv2.imshow('BEV', BEV_img)
# # cv2.imwrite(r'C:\Users\user\Ponc\terrinus\grid.jpeg', img)
# # cv2.imwrite(r'C:\Users\user\Ponc\terrinus\grid_BEV.jpeg', BEV_img)
# cv2.waitKey(0)

squares = []
h,w,c = BEV_img.shape
delta_x = np.int(h/(num_x-1))
delta_y = np.int(w/(num_y-1))
print("Image shape: ", h,w,c)
print("Delta x: ",delta_x," delta_y = ",delta_y)
occ = np.ones((num_x-1,num_y-1))
occ[0,0] = 0
occ[0,1] = 0
occ[1,0] = 0
occ[2,0] = 0
occ[3,0] = 0
occ[1,1] = 0
candidates_each_row = []
for i in range(num_x-1):
    candidates = 0
    x_mid = 0
    y_mid = 0
    for j in range(num_y-1):
        print("i,j = ", i,j)
        print("j*delta_y = ",j*delta_y)
        square = BEV_img[i*delta_x:i*delta_x+delta_x,j*delta_y:j*delta_y+delta_y,:]
        # process square
        if(occ[i,j]==1):
            mean_x = np.mean(np.array([i*delta_x, i*delta_x+delta_x]))
            mean_y = np.mean(np.array([j*delta_y, j*delta_y+delta_y]))
            candidates = candidates + 1
            x_mid = x_mid + mean_x
            y_mid = y_mid + mean_y
    if(candidates==0):
        candidates_each_row.append(-1)
    else:
        candidates_each_row.append((x_mid/candidates, y_mid/candidates))

for c in candidates_each_row:
    # theres num_x-1 of these
    if(c!=-1):
        cv2.circle(BEV_img,(int(c[1]), int(c[0])), 5, (0,0,255))
cv2.imwrite("./../imgs/grid_BEV_trajectory.jpeg", BEV_img)
cv2.imshow('asdf', BEV_img)
cv2.waitKey(0)

def get_square_points_img(qs, nqx, nqy,poins_grid_img):
    """
    AUX/DEBUG function to draw overlays
    qs : tuple containing (q_x, q_y)
    returns the four image points corresponding to the square's corners 2x4
    """
    q_x, q_y = qs
    square_corners = np.zeros((2,4))
    for i in range(4):
        a_x, a_y = np.unravel_index(i, (2,2))
        index_in_grid_matrix = (q_x + a_x)*(nqy + 1) + (q_y + a_y)
        square_corners[:, i] = poins_grid_img[:,index_in_grid_matrix]
    return square_corners

candidates_each_row = []
for i in range(num_x-1):
    print(i, "-th ROW")
    mitja_x = 0
    candidates = 0
    mitja_y = 0
    for j in range(num_y-1):
        if(occ[i,j]==1):
            spoints = get_square_points_img((i,j),num_x-1,num_y-1,points_grid_img)
            rata,ta = np.mean(spoints,axis=1)
            print(rata,ta)
            candidates = candidates + 1
            mitja_x = mitja_x + rata
            mitja_y = mitja_y + ta
    candidates_each_row.append((mitja_x/candidates,mitja_y/candidates))
print(candidates_each_row)
for c in candidates_each_row:
    # theres num_x-1 of these
    if(c[0]!=0.0):
        cv2.circle(img,(int(c[0]), int(c[1])), 7, (0,0,255))
cv2.imshow('asdffdf', img)
cv2.waitKey(0)
cv2.imwrite("./../imgs/grid_img_trajectory.jpeg", BEV_img)