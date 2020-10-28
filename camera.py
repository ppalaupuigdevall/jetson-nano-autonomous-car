import numpy as np
import cv2
from jetcam.csi_camera import CSICamera
from PIL import Image

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



class Camera:

    def __init__(self, height = 480, width = 640):
        # Intrinsics results obtained from calib module
        self._K = np.array([[676.74,0,317.4],[0,863.29,252.459],[0,0,1]])
        # Extrinisics
        angles = np.array([90,90,0])*np.pi/180
        R = eulerAnglesToRotationMatrix(angles) # Rotation matrix  WCS->CCS
        T = np.array([[0.0],[-0.156],[0.0]]) # WCS expressed in camera coordinate system 
        self._Rt = np.hstack((R,T))
        self._camera = CSICamera(width, height)
        print("Camera Initialized - Height = {}, Width = {}".format(height, width))

    def get_Rt(self):
        return self._Rt
    
    def get_K(self):
        return self._K

    def get_frame_cv2(self):
        """
        Returns an ndarray in BGR format (OpenCV format)
        """
        return self._camera.read()

    def get_frame_PIL(self):
        """
        Returns a Image object in RGB (PIL format)
        """
        image = self._camera.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        return image