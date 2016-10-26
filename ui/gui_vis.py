import numpy as np
import cv2
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from . import save_result
from lib import utils


class GUI_VIS(QWidget):
    def __init__(self, opt_engine, topK=16, grid_size=None, nps=320, model_name='tmp'):
        QWidget.__init__(self)
        self.topK = topK
        if grid_size is None:
            self.n_grid = int(np.ceil(np.sqrt(self.topK)))
            self.grid_size = (self.n_grid, self.n_grid) # (width, height)
        else:
            self.grid_size = grid_size
        self.select_id = 0
        self.ims = None
        self.vis_results = None
        self.width = int(np.round(nps/ (4 * float(self.grid_size[1])))) * 4
        self.winWidth = self.width * self.grid_size[0]
        self.winHeight = self.width * self.grid_size[1]

        self.setFixedSize(self.winWidth, self.winHeight)
        self.opt_engine = opt_engine
        self.frame_id = -1
        self.sr = save_result.SaveResult(model_name=model_name)

    def save(self):
        self.sr.cache_result(ims=self.ims, visIms=self.vis_results)
        self.sr.save()

    def set_frame_id(self, frame_id):
        if frame_id != self.frame_id:
            self.frame_id = frame_id
            self.update_vis()

    def set_image_id(self, image_id):
        if image_id != self.select_id:
            self.select_id = image_id
            self.update_vis()

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(event.rect(), Qt.white)
        if self.vis_results is not None:
            qImg = QImage(self.vis_results.tostring(), self.winWidth, self.winHeight, QImage.Format_RGB888)
            painter.drawImage(0, 0, qImg)

        painter.end()

    def update_results(self, ims):
        self.ims = ims

    def mousePressEvent(self, event):
        pos = event.pos()
        if event.button() == Qt.LeftButton:
            x_select = np.floor(pos.x() / float(self.width))
            y_select = np.floor(pos.y() / float(self.width))
            new_id = int(y_select * self.grid_size[0] + x_select)
            print('pos=(%d,%d) (x,y)=(%d,%d) image_id=%d' % (int(pos.x()), int(pos.y()), x_select, y_select, new_id))
            if new_id != self.select_id:
                self.select_id = new_id
                self.update_vis()
                self.update()
                self.emit(SIGNAL('update_image_id'), self.select_id)

    def sizeHint(self):
        return QSize(self.winWidth, self.winHeight)

    def reset(self):
        self.ims = None
        self.vis_results = None
        self.update()
        self.sr.reset()
        self.frame_id = -1
        self.select_id=0

    def get_show_id(self):
        return self.select_id

    def set_show_id(self, _id):
        if _id != self.select_id:
            self.select_id = _id
            self.show_results()

    def update_vis(self):
        ims = self.opt_engine.get_images(self.frame_id)

        if ims is not None:
            self.ims = ims

        if self.ims is None:
            return

        ims_show = []
        n_imgs = self.ims.shape[0]
        for n in range(n_imgs):
            # im = ims[n]
            im_s = cv2.resize(self.ims[n], (self.width, self.width), interpolation=cv2.INTER_CUBIC)
            if n == self.select_id and self.topK > 1:
                t = 3  # thickness
                cv2.rectangle(im_s, (t, t), (self.width - t, self.width - t), (0, 255, 0), t)
            im_s = im_s[np.newaxis, ...]
            ims_show.append(im_s)
        if ims_show:
            ims_show = np.concatenate(ims_show, axis=0)
            g_tmp = utils.grid_vis(ims_show, self.grid_size[1], self.grid_size[0]) # (nh, nw)
            self.vis_results = g_tmp.copy()
            self.update()
