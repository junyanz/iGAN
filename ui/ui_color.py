import numpy as np

import cv2


class UIColor:
    def __init__(self, img_size, scale, nc=3):
        self.img_size = img_size
        self.scale = float(scale)
        self.nc = nc
        self.img = np.zeros((img_size, img_size, self.nc), np.uint8)
        self.mask = np.zeros((img_size, img_size, 1), np.uint8)
        self.width = int(2*scale)


    def update(self, points, color):
        num_pnts = len(points)
        w = int(max(1, self.width / self.scale))
        c = (color.red(), color.green(), color.blue())
        for i in range(0, num_pnts - 1):
            pnt1 = (int(points[i].x() / self.scale), int(points[i].y() / self.scale))
            pnt2 = (int(points[i + 1].x() / self.scale), int(points[i + 1].y() / self.scale))
            if self.nc == 3:
                cv2.line(self.img, pnt1, pnt2, c, w)
            else:
                cv2.line(self.img, pnt1, pnt2, c[0], w)
            cv2.line(self.mask, pnt1, pnt2, 255, w)


    def get_constraints(self):
        return self.img, self.mask

    def update_width(self, d):
        self.width = min(20, max(1, self.width+ d))
        return self.width

    def reset(self):
        self.img = np.zeros((self.img_size, self.img_size, self.nc), np.uint8)
        self.mask = np.zeros((self.img_size, self.img_size, 1), np.uint8)
