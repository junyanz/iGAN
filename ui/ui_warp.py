import numpy as np
import cv2
from lib import utils

class UIWarp():
    def __init__(self, img_size, scale, nc=3):
        self.img_size = img_size
        self.scale = scale
        self.nc = nc
        self.img = np.zeros((img_size, img_size, self.nc), np.uint8)
        self.mask = np.zeros((img_size, img_size, 1), np.uint8)
        self.init_width = 24
        self.width = int(self.init_width * self.scale)
        self.points1 = []
        self.points2 = []
        self.widths  = []
        self.ims = []
        self.activeId = -1
        self.im = None

    def CropPatch(self, pnt, width):
        [x_c,y_c] = pnt
        w =  width / 2.0
        x1 = int(np.clip(x_c - w, 0, self.img_size - 1))
        y1 = int(np.clip(y_c - w, 0, self.img_size - 1))
        x2 = int(np.clip(x_c + w, 0, self.img_size - 1))
        y2 = int(np.clip(y_c + w, 0, self.img_size - 1))
        return [x1,y1,x2,y2]

    def AddPoint(self, pos, im):
        x_c = int(np.round(pos.x()/self.scale))
        y_c=  int(np.round(pos.y()/self.scale))
        pnt = (x_c, y_c)
        print('add point (%d,%d)' % pnt)
        self.points1.append(pnt)
        self.points2.append(pnt)
        self.widths.append(self.width)

        self.im = cv2.resize(im, (self.img_size, self.img_size))#*255).astype(np.uint8)
        self.ims.append(self.im.copy())
        self.activeId = len(self.points1) - 1
        print('set activeId =%d' % self.activeId)


    def StartPoint(self):
        print('start point: activeId = %d' % self.activeId)
        if self.activeId >= 0 and self.points1:
            return self.points1[self.activeId]
        else:
            return None

    def update(self, pos):
        self.img = np.zeros((self.img_size, self.img_size, self.nc), np.uint8)
        self.mask = np.zeros((self.img_size, self.img_size, 1), np.uint8)

        print('uiWarp: update %d' % self.activeId)
        if self.activeId >= 0:
            x_c = int(np.round(pos.x()/self.scale))
            y_c=  int(np.round(pos.y()/self.scale))
            pnt = (x_c, y_c)
            self.points2[self.activeId] = pnt

        count = 0
        for pnt1, pnt2 in zip(self.points1, self.points2):
            w = int(max(1, self.width / self.scale))
            [x1,y1,x2,y2] = self.CropPatch(pnt1, w)
            [x1n, y1n, x2n, y2n] = self.CropPatch(pnt2, w)
            im = self.ims[count]
            if self.nc == 3:
                patch = im[y1:y2,x1:x2,:].copy()
                self.img[y1n:y2n,x1n:x2n,:] = patch
                self.mask[y1n:y2n,x1n:x2n,:] = 255
            else:
                patch = im[y1:y2, x1:x2, [0]].copy()
                self.img[y1n:y2n, x1n:x2n] = patch
                self.mask[y1n:y2n, x1n:x2n] = 255
            count += 1


    def get_constraints(self):
        return self.img, self.mask

    def get_edge_constraints(self):
        return self.img, self.mask

    def update_width(self, d):
        self.width = min(256, max(32, self.width + d * 4 * self.scale))
        if self.activeId >= 0:
            self.widths[self.activeId] = self.width
            print('update width %d, activeId =%d'%(self.width, self.activeId))
        return self.width

    def reset(self):
        self.activeId = -1
        self.points1 = []
        self.points2 = []
        self.widths = []
        self.ims = []
        self.width = int(self.init_width * self.scale)
        self.img = np.zeros((self.img_size, self.img_size, self.nc), np.uint8)
        self.mask = np.zeros((self.img_size, self.img_size, 1), np.uint8)
