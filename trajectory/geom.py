import numpy as np
K = np.array([[676.74,0,317.4],[0,863.29,252.459],[0,0,1]])
Rt = np.array([[0,-1,0,0.0],[0,0,-1,0.0],[1,0,0,0.15]])

Xw = np.array([[0.5],[0.2],[0.0],[1.0]])
xworld = 0.5
yworld = -0.5
lengthx = 1
lengthy = 1

# TODO: Define this according to lengthx and lengthy
Xw = np.array([[xworld + lengthx, xworld + lengthx, xworld, xworld          ], \
               [yworld + lengthy, yworld          , yworld, yworld + lengthy], \
               [0.0             , 0.0             , 0.0   , 0.0             ],\
               [1.0             , 1.0             , 1.0   , 1.0             ]])

print(Xw)
xp = np.zeros((2,4))
for i in range(4):
    pcc = np.matmul(Rt,Xw[:,i])
    ximg = np.matmul(K, pcc)
    # print(ximg)
    ximg = ximg[:2]/ximg[2]
    print("Point in image = (", int(ximg[0]), ", ", int(ximg[1]), ")")     
