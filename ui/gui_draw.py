import numpy as np
import time
import cv2
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from lib import utils
# from scipy import ndimage
from .ui_recorder import UIRecorder
from .ui_color import UIColor
from .ui_sketch import UISketch
from .ui_warp import UIWarp

class GUIDraw(QWidget):
    def __init__(self, opt_engine, win_size=320, img_size=64, topK=16, useAverage=False, shadow=False):
        QWidget.__init__(self)
        self.isPressed = False
        self.points = []
        self.topK = topK
        self.shadow = False
        self.lastDraw = 0
        self.model = None
        self.shadow = shadow
        self.init_color(shadow)
        self.opt_engine = opt_engine
        self.pos = None
        self.nps = win_size
        self.scale = win_size / float(img_size)
        self.brushWidth = int(2 * self.scale)
        self.show_nn = True
        self.type = 'edge' if self.shadow else 'color'
        self.show_ui = True
        self.uir = UIRecorder(shadow=shadow)
        nc = 1 if shadow else 3
        self.uiColor = UIColor(img_size=img_size, scale=self.scale, nc=nc)
        self.uiSketch = UISketch(img_size=img_size, scale=self.scale, nc=nc)
        self.uiWarp = UIWarp(img_size=img_size, scale=self.scale, nc=nc)
        self.img_size = img_size
        self.move(win_size, win_size)
        self.useAverage = useAverage
        if self.shadow:
            self.setMouseTracking(True)
        self.movie = True
        self.frame_id = -1
        self.image_id = 0

    def change_average_mode(self):
        self.useAverage = not self.useAverage
        self.update()

    def update_opt_engine(self):
        if self.type in ['color', 'edge']:
            [im_c, mask_c] = self.uiColor.get_constraints()
            [im_e, mask_e] = self.uiSketch.get_constraints()
        else:
            [im_c, mask_c] = self.uiWarp.get_constraints()
            [im_e, mask_e] = self.uiWarp.get_edge_constraints()

        self.opt_engine.set_constraints([im_c, mask_c, im_e, mask_e])
        self.opt_engine.update()
        self.frame_id = -1

    def update_im(self):
        self.update()
        QApplication.processEvents()

    def update_ui(self):
        if self.opt_engine.is_fixed():
            self.set_frame_id(-1)
            self.set_image_id(0)
            self.emit(SIGNAL('update_image_id'), 0)
            self.opt_engine.update_fix()
        if self.type is 'color':
            self.uiColor.update(self.points, self.color)
        if self.type is 'edge':
            self.uiSketch.update(self.points, self.color)
        if self.type is 'warp':
            self.uiWarp.update(self.pos)

    def set_image_id(self, image_id):
        if self.image_id != image_id:
            self.image_id = image_id
            self.update()

    def set_frame_id(self, frame_id):
        if self.frame_id != frame_id:
            self.frame_id = frame_id
            self.update()


    def reset(self):
        self.isPressed = False
        self.points = []
        self.lastDraw = 0
        self.uir.reset()
        self.uiSketch.reset()
        self.uiColor.reset()
        self.uiWarp.reset()
        self.frame_id = -1
        self.image_id = 0

        self.update()

    def round_point(self, pnt):
        # print(type(pnt))
        x = int(np.round(pnt.x()))
        y = int(np.round(pnt.y()))
        return QPoint(x, y)

    def init_color(self, shadow):
        if shadow:
            self.color = QColor(0, 0, 0)  # shadow mode: default color black
        else:
            self.color = QColor(0, 255, 0)  # default color red
        self.prev_color = self.color

    def change_color(self):
        if self.shadow:
            if self.color == QColor(0, 0, 0):
                self.color = QColor(255, 255, 255)
            else:
                self.color = QColor(0, 0, 0)
        else:
            color = QColorDialog.getColor(parent=self)
            self.color = color

        self.prev_color = self.color
        self.emit(SIGNAL('update_color'), QString('background-color: %s' % self.color.name()))


    def get_image_id(self):
        return self.image_id

    def get_frame_id(self):
        return self.frame_id

    def get_z(self):
        print('get z from image %d, frame %d'%(self.get_image_id(), self.get_frame_id()))
        return self.opt_engine.get_z(self.get_image_id(), self.get_frame_id())

    def shadow_image(self, img, pos):
        if img is None:
            return None
        weighted_img = np.ones((img.shape[0], img.shape[1]), np.uint8)
        x = int(pos.x() / self.scale)
        y = int(pos.y() / self.scale)

        weighted_img[y, x] = 0
        dist_img = cv2.distanceTransform(weighted_img, distanceType=cv2.cv.CV_DIST_L2, maskSize=5).astype(np.float32)
        dist_sigma = self.img_size/2.0
        dist_img_f = np.exp(-dist_img / dist_sigma)
        dist_img_f = np.tile(dist_img_f[..., np.newaxis], [1,1,3])
        l = 0.25
        img_f = img.astype(np.float32)
        rst_f = (img_f * l + (1-l) * (img_f * dist_img_f + (1-dist_img_f)*255.0))
        rst = rst_f.astype(np.uint8)
        return rst

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.fillRect(event.rect(), Qt.white)
        painter.setRenderHint(QPainter.Antialiasing)
        im = self.opt_engine.get_image(self.get_image_id(), self.get_frame_id(), self.useAverage)
        if self.shadow and self.useAverage:
            im = self.shadow_image(im, self.pos)

        if im is not None:
            bigim = cv2.resize(im, (self.nps, self.nps))
            qImg = QImage(bigim.tostring(), self.nps, self.nps, QImage.Format_RGB888)
            painter.drawImage(0, 0, qImg)

        # draw path
        if self.isPressed and self.type in ['color', 'edge'] and self.show_ui:
            if self.type is 'edge':
                if self.shadow:
                    painter.setPen(QPen(self.color, 10,  cap=Qt.RoundCap, join=Qt.RoundJoin))
                else:
                    painter.setPen(QPen(Qt.gray, 10, Qt.DotLine, cap=Qt.RoundCap, join=Qt.RoundJoin))
            else:
                painter.setPen(QPen(self.color, int(self.brushWidth), cap=Qt.RoundCap, join=Qt.RoundJoin))

            n_pnts = len(self.points)
            for i in range(0, n_pnts-5, 5):
                painter.drawLine(self.points[i], self.points[i + 5])
            self.lastDraw = n_pnts

        # draw cursor
        if self.pos is not None:
            w = self.brushWidth
            c = self.color
            ca = QColor(255, 255, 255, 127)
            pnt = QPointF(self.pos.x(), self.pos.y())
            if self.type is 'color':
                ca = QColor(c.red(), c.green(), c.blue(), 127)
            if self.type is 'edge':
                ca = QColor(0, 0, 0, 127)
            if self.type is 'warp':
                ca = QColor(0, 0, 0, 127)

            painter.setPen(QPen(ca, 1))
            painter.setBrush(ca)
            if self.type is 'warp':
                if self.show_ui:
                    painter.drawRect(int(self.pos.x()-w/2.0),int(self.pos.y() - w/2.0), w, w)
            else:
                painter.drawEllipse(pnt, w, w)

        if self.type is 'warp' and self.show_ui:
            color = Qt.green
            w = 10
            painter.setPen(QPen(color, w, Qt.DotLine, cap=Qt.RoundCap, join=Qt.RoundJoin))  # ,)
            pnt1 = self.uiWarp.StartPoint()
            if pnt1 is not None:
                pnt1f = QPointF(pnt1[0]*self.scale, pnt1[1]*self.scale)
                pnt2f = QPointF(self.pos.x(), self.pos.y())
                painter.drawLine(pnt1f, pnt2f)

        if self.show_ui:
            self.uir.draw(painter)
        painter.end()

    def update_msg(self, painter):
        # msgs = []
        if self.type is 'color':
            msg = 'coloring: (%d, %d, %d)' % (self.color.red(), self.color.green(), self.color.blue())
        if self.type is 'edge':
            msg = 'sketching'

        if self.type is 'warp':
            msg = 'warping'

        painter.setPen(QColor(0, 0, 0))
        fontSz = 10
        border = 3
        painter.setFont(QFont('Decorative', fontSz))
        painter.drawText(QPoint(border, fontSz + border), QString(msg))
        num_frames = self.opt_engine.get_num_frames()
        num_images = self.opt_engine.get_num_images()
        if num_frames > 0 and num_images > 0:
            d_frame_id = (self.get_frame_id())%num_frames + 1
            d_show_id = (self.get_image_id())% num_images + 1
            msg = 'frame %2d/%2d, image %2d/%2d'%(d_frame_id, num_frames, d_show_id, num_images)
            painter.setPen(QColor(0, 0, 0))
            fontSz = 10
            border = 3
            painter.setFont(QFont('Decorative', fontSz))
            painter.drawText(QPoint(border, 2 * fontSz + border), QString(msg))

    def wheelEvent(self, event):
        d = event.delta() / 120
        if self.type is 'edge':
            self.brushWidth = self.uiSketch.update_width(d, self.color)
        if self.type is 'color':
            self.brushWidth = self.uiColor.update_width(d)
        if self.type is 'warp':
            self.brushWidth = self.uiWarp.update_width(d)
        self.update()

    def mousePressEvent(self, event):
        self.pos = self.round_point(event.pos())

        if event.button() == Qt.LeftButton:
            self.isPressed = True
            self.points.append(self.pos)
            self.update_opt_engine()
            self.update_ui()
            self.update()

        if event.button() == Qt.RightButton:
            if self.type in ['edge', 'color']:# or self.type is 'edge':
                self.change_color()

            if self.type is 'warp':
                im = self.opt_engine.get_image(self.get_image_id(), self.get_frame_id())
                self.uiWarp.AddPoint(event.pos(), im)
                self.brushWidth = self.uiWarp.update_width(0)
            self.update()

    def mouseMoveEvent(self, event):
        self.pos = self.round_point(event.pos())
        if self.isPressed:
            if self.type in ['color','edge']:
                self.points.append(self.pos)
            self.update_ui()
            self.update_opt_engine()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.isPressed:
            self.update()
            if self.type is 'color' or self.type is 'edge':
                self.uir.save_record(self.points, self.color, self.brushWidth, self.type)

            self.opt_engine.save_constraints()
            self.uiColor.reset()
            self.uiSketch.reset()
            self.uiWarp.reset()

            del self.points[:]
            self.isPressed = False
            self.lastDraw = 0

    def sizeHint(self):
        return QSize(self.nps, self.nps)  # 28 * 8


    def update_frame(self, dif):
        num_frames = self.opt_engine.get_num_frames()
        if num_frames > 0:
            self.frame_id = (self.frame_id+dif) % num_frames
            print("show frame id = %d"%self.frame_id)

    def fix_z(self):
        self.opt_engine.init_z(self.get_frame_id(), self.get_image_id())

    def morph_seq(self):
        self.frame_id=0
        num_frames = self.opt_engine.get_num_frames()
        print('show %d frames' % num_frames)
        for n in range(num_frames):
            self.update()
            QApplication.processEvents()
            fps = 10
            time.sleep(1/float(fps))
            self.emit(SIGNAL('update_frame_id'),self.frame_id)
            if n < num_frames-1: # stop at last frame
                self.update_frame(1)

    def use_color(self):
        print('coloring')
        self.type = 'color'
        self.color = self.prev_color
        self.emit(SIGNAL('update_color'), QString('background-color: %s' % self.color.name()))
        self.brushWidth = self.uiColor.update_width(0)
        self.update()

    def use_edge(self):
        print('sketching')
        self.type = 'edge'
        self.color = QColor(0, 0, 0) if self.shadow else QColor(128, 128, 128)
        self.emit(SIGNAL('update_color'), QString('background-color: %s' % self.color.name()))
        self.brushWidth = self.uiSketch.update_width(0, self.color)
        self.update()

    def use_warp(self):
        self.type = 'warp'
        self.color = QColor(128, 128, 128)
        self.emit(SIGNAL('update_color'), QString('background-color: %s' % self.color.name()))
        self.brushWidth = self.uiWarp.update_width(0)
        print('warp brush: %d' % self.brushWidth)
        self.update()


    def show_edits(self):
        self.show_ui = not self.show_ui
        self.update()

