import numpy as np

import cv2
from PyQt4.QtCore import *


# import HOGNet

class UISketch:
    def __init__(self, npx, scale, accu=True):
        self.npx = npx
        self.scale = scale
        self.nc = 3
        self.img = np.zeros((npx, npx, self.nc), np.uint8)
        self.mask = np.zeros((npx, npx, 1), np.uint8)
        self.width = 1
        # self.accu = True

    def update(self, points, color):
        n_pnts = len(points)
        if color is Qt.black:
            print('add sketching')
            c = 255
        if color is Qt.white:
            print('remove gradients')
            c = 0
        for i in range(0, n_pnts - 1):
            pnt1 = (points[i].x()/self.scale, points[i].y()/self.scale)
            pnt2 = (points[i + 1].x()/self.scale, points[i + 1].y()/self.scale)
            cv2.line(self.img, pnt1, pnt2, (c,c,c), self.width)
            cv2.line(self.mask, pnt1, pnt2, 255, self.width)
        # utils.CVShow(self.img, 'sketch input image')
        # utils.CVShow(self.mask, 'sketch image mask')
        # cv2.imwrite('sketch.png', self.img)
        # cv2.imwrite('mask.png', self.mask)


    def update_width(self, d, color):
        if color is Qt.white:
            self.width = min(20, max(1, self.width+ d))
        if color is Qt.black:
            self.width = 1
        return self.width

    def get_constraints(self):
        return self.img, self.mask
        # img = self.img[np.newaxis, :]
        # mask = self.mask[np.newaxis, :]
        # img_t = modeldef.transform_mask(img)
        # mask_t = modeldef.transform_mask(mask)
        # hog_mask = HOGNet.comp_mask(mask_t)
        # return img_t, hog_mask

    def reset(self):
        self.img = np.zeros((self.npx, self.npx, self.nc), np.uint8)
        self.mask = np.zeros((self.npx, self.npx, 1), np.uint8)


