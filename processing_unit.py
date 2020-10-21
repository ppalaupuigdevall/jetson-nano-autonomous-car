import numpy as np
import cv2

class GeomPU:
    def __init__(self, K, Rt, grid_params):
        """
        Rt: [ R | t] extrinsinc parameters of the camera
        grid_params: dictionary containing {xw,yw,length_x, length_y, num_x, num_y}
        """
        self._K = K
        self._Rt = Rt
        self._xw = grid_params['xw']
        self._yw = grid_params['yw']
        self._length_x = grid_params['length_x']
        self._length_y = grid_params['length_y']
        self._num_x = grid_params['num_x'] # Number of vertical vertices of the grid
        self._num_y = grid_params['num_y'] # Number of horizontal vertices of the grid
        self._nqx = self._num_x - 1 # Number of squares per column
        self._nqy = self._num_y - 1 # number of squares per row
        self._grid_points = self.ccs_to_img_coord(self._wcs_to_ccs(self.generate_grid()))
        self._four_points_BEV = np.array([[self._xw + self._length_x, self._xw + self._length_x, self._xw,                 self._xw],\
                                          [self._yw,                  self._yw - self._length_y, self._yw - self.length_y, self._yw],\
                                          [0.0             ,          0.0             ,          0.0   ,                   0.0     ],\
                                          [1.0             ,          1.0             ,          1.0   ,                   1.0     ]])
    
        # Create homography for BEV
        self.get_four_point_transform(self._four_points_BEV)

    def generate_grid_world(self):
        # Generates a self._num_x * self._num_y grid in real world coordinates
        grid_points = np.zeros((4,self._num_x*self._num_y))
        grid_points[3,:] = 1.0
        for a in range(self._num_x * self._num_y):
            i = a//self._num_y
            j = a%self._num_y
            x = self._xw + self._length_x - ((i/(self._num_x-1))*self._length_x)
            y = self._yw - ((j/(self._num_y-1))*self._length_y)
            grid_points[0, a] = x
            grid_points[1, a] = y
        return grid_points

    def wcs_to_ccs(self, X_wcs):
        """
        X_w: 4xN matrix containing points in WCS in homogeneous coordinates
        """
        Pccs = np.matmul(self._Rt, X_w)
        return Pccs
    
    def ccs_to_img_coord(self, X_ccs):
        """
        X_ccs: 4xN matrix contatining points in CCS
        returns 2xN matrix containing image coordinates
        """
        Pimg_coord = np.matmul(self._K, X_ccs) 
        Pimg_coord = Pimg_coord[:2,:]/Pimg_coord[2,:]
        return Pimg_coord

    def get_upper_left_coord(self, q):
        """
        AUX/DEBUG function to draw overlays
        q is the number of square [0, ..., nqx*nqy -1]
        returns the upper left coordinate of the square, which coincides with grid coordinates
        """
        q_x = q//self._nqy
        q_y = q%self._nqy
        return np.array([q_x, q_y])

    def evaluate_grid(self, BEV_img):
        h,w,c = BEV_img.shape
        delta_x = np.int(h/self._nqx)
        delta_y = np.int(w/self._nqy)
        throttle_map = np.zeros((self._num_x))
        trajectory_BEV = np.zeros((2,self._num_x))
        trajectory_img = np.zeros((2,self._num_x))

        for i in range(self._num_x):
            # get average coordinates for each row
            cands = 0
            x_mid_BEV = 0
            y_mid_BEV = 0
            x_mid_img = 0
            y_mid_img = 0
            for j in range(self._num_y):
                value = BEV_img[i*delta_x, j*delta_y]
                if value == self._INDEX_FLOOR:
                    cands = cands + 1
                    throttle_map[i] = throttle_map[i] + 1
                    x_mid_BEV = x_mid_BEV + i*delta_x
                    y_mid_BEV = y_mid_BEV + j*delta_y
                    # img points 
                    index_in_grid_matrix = (i*self._num_y) + j
                    point = self._grid_points[:,index_in_grid_matrix]
                    x_mid_img = x_mid_img + point[0]
                    y_mid_img = y_mid_img + point[1]
            
            trajectory_BEV[0,i] = x_mid_BEV/cands
            trajectory_BEV[1,i] = y_mid_BEV/cands
            trajectory_img[0,i] = x_mid_img/cands
            trajectory_img[1,i] = y_mid_img/cands
        
        return throttle_map, trajectory_BEV, trajectory_img


    def get_square_points_img(self, qs):
        """
        SQUARES MODE Evaluates squares instead of grid points
        qs : tuple containing (q_x, q_y)
        returns the four image points corresponding to the square's corners 2x4
        """
        q_x, q_y = qs
        square_corners = np.zeros((2,4))
        for i in range(4):
            a_x, a_y = np.unravel_index(i, (2,2))
            index_in_grid_matrix = (q_x + a_x)*(self._nqy + 1) + (q_y + a_y)
            square_corners[:, i] = self._grid_points[:,index_in_grid_matrix]
        return square_corners

    def get_squares(self, BEV_img):
        """
        SQUARES MODE 
        """
        squares = []
        h,w,c = BEV_img.shape
        delta_x = np.int(h/self._nqx)
        delta_y = np.int(w/self._nqy)
        for i in range(self._nqx):
            for j in range(self._nqy):
                square = BEV_img[i*delta_x:i*delta_x+delta_x,j*delta_y:j*delta_y+delta_y,:]
                squares.append(squares)
        return squares
    
    def process_square(self, square):
        ret,thresh = cv2.threshold(square, 0.5,1.5, cv2.THRESH_BINARY_INV)
        
    
    
    def get_four_point_transform(self, pts):
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
        H = cv2.getPerspectiveTransform(np.float32(rect), np.float32(dst))
        # warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        # return the warped image
        self._H = H
        self._maxWidth = maxWidth
        self._maxHeight = maxHeight

    def create_BEV_image(self, image):
        warped = cv2.warpPerspective(image, self._H, (self._maxWidth, self._maxHeight))
        return warped


class ProcessingUnit:
    def __init__(self):
        print("")
