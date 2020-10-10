import numpy as np
import cv2

class GeomPU:
    def __init__(self, Rt, grid_params):
        """
        Rt: [ R | t] extrinsinc parameters of the camera
        grid_params: dictionary containing {xw,yw,length_x, length_y, num_x, num_y}
        """
        self._Rt = Rt
        # Grid is a matrix (4, num_x*num_y) elements containing the grid points
        self._xw = grid_params['xw']
        self._yw = grid_params['yw']
        self._length_x = grid_params['length_x']
        self._length_y = grid_params['length_y']
        self._num_x = grid_params['num_x']
        self._num_y = grid_params['num_y']
        self._grid_points = self.generate_grid()
        self._four_points_BEV = np.array([[self._xw + self._length_x, self._xw + self._length_x, self._xw,                 self._xw],\
                                          [self._yw,                  self._yw - self._length_y, self._yw - self.length_y, self._yw],\
                                          [0.0             ,          0.0             ,          0.0   ,                   0.0     ],\
                                          [1.0             ,          1.0             ,          1.0   ,                   1.0     ]])
    
    def generate_grid(self):
        # TODO generate grid wirh code in GEOM
        grid_points = np.zeros((4,self._num_x*self._num_y))
        grid_points[3,:] = 1.0
        for a in range(self._num_x * self._num_y):
            grid_points[0, a] = self._xw + self._length_x - ((a//self._num_y)/self._num_x)*self._length_x
            grid_points[1, a] = self._yw + self._length_y - ((a//self._num_x)/self._num_y)*self._length_y
        return grid_points

    def four_point_transform(self, image, pts):
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
        M = cv2.getPerspectiveTransform(np.float32(rect), np.float32(dst))
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        # return the warped image
        return warped
    

    def create_BEV_image(self, img):
        """
        img must be in opencv format
        """
        BEV_img = self.four_point_transform(img, self._four_points_BEV)
        return BEV_img


class ProcessingUnit:
    def __init__(self):
        
