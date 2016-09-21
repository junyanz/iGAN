import numpy as np
import time
import cv2
from PyQt4.QtCore import *
from PyQt4.QtGui import *

from ui_recorder import UIRecorder
from ui_color import UIColor
from ui_sketch import UISketch
from ui_warp import UIWarp
# from backend import model_config

class GUIDraw(QWidget):
    def __init__(self, opt_engine, batch_size=32, n_iters=25, nps=320, topK=16, hash=None, hash_z=None):
        QWidget.__init__(self)
        self.isPressed = False
        self.points = []
        self.topK = topK
        self.lastDraw = 0
        self.model = None
        self.color = QColor(0, 0, 0)  # default black
        self.opt_engine = opt_engine
        npx = opt_engine.npx
        self.pos = None
        self.nps = nps
        self.scale = nps / npx
        self.brushWidth = int(2 * self.scale)
        self.show_nn = True
        self.type = 'color'
        self.show_ui = True
        self.uir = UIRecorder()

        self.uiColor = UIColor(npx=npx, scale=self.scale)
        self.uiSketch = UISketch(npx=npx, scale=self.scale)
        self.uiWarp = UIWarp(npx=npx, scale=self.scale)
        self.move(nps, nps)

        self.movie = True
        self.frame_id = -1
        self.image_id = 0

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
        if self.opt_engine.just_fixed:
            self.set_frame_id(-1)
            self.set_image_id(0)
            self.emit(SIGNAL('update_image_id'), 0)
            self.opt_engine.just_fixed=False
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
        x = np.round(pnt.x() / self.scale) * self.scale
        y = np.round(pnt.y() / self.scale) * self.scale
        return QPoint(x, y)

    def update_beta(self, step):
        beta_r = self.model[-1]
        value = beta_r.get_value()
        print('<before> beta=%5.5f' % beta_r.get_value())
        value *= step
        beta_r.set_value(value)
        print('<after> beta=%5.5f' % beta_r.get_value())
        # self.generate()
        self.update()

    def get_image_id(self):
        return self.image_id
        # return self.visNN.get_show_id()

    def get_frame_id(self):
        return self.frame_id
    def get_z(self):
        print('get z from image %d, frame %d'%(self.get_image_id(), self.get_frame_id()))
        return self.opt_engine.get_z(self.get_image_id(), self.get_frame_id())

    def paintEvent(self, event):
        # print('paintEvent')
        # print('update paint %d'%self.frame_id)
        # self.paint_im = QImage(self.nps, self.nps, QImage.Format_RGB888)

        painter = QPainter()
        painter.begin(self)
        painter.fillRect(event.rect(), Qt.white)
        painter.setRenderHint(QPainter.Antialiasing)
        im = self.opt_engine.get_image(self.get_image_id(), self.get_frame_id())

        if im is not None:
            bigim = cv2.resize(im, (self.nps, self.nps))
            qImg = QImage(bigim.tostring(), self.nps, self.nps, QImage.Format_RGB888)
            painter.drawImage(0, 0, qImg)

        # draw path
        if self.isPressed and self.type in ['color', 'edge'] and self.show_ui:
            if self.type is 'edge':
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
                if self.color is Qt.white:
                    ca = QColor(255, 255, 255, 127)
                if self.color is Qt.black:
                    ca = QColor(0, 0, 0, 127)
            if self.type is 'warp':
                ca = QColor(0, 0, 0, 127)

            painter.setPen(QPen(ca, 1))
            painter.setBrush(ca)
            if self.type is 'warp':
                if self.show_ui:
                    painter.drawRect(int(self.pos.x()-w/2),int(self.pos.y()-w/2), w, w)
            else:
                painter.drawEllipse(pnt, w, w)

        if self.type is 'warp' and self.show_ui:
            # print 'paint warp brush'
            color = Qt.green
            w = 10
            painter.setPen(QPen(color, w, Qt.DotLine, cap=Qt.RoundCap, join=Qt.RoundJoin))  # ,)
            pnt1 = self.uiWarp.StartPoint()
            # print 'start_point', pnt1
            if pnt1 is not None:
                # print 'paint warp brush 2'
                pnt1f = QPointF(pnt1[0]*self.scale, pnt1[1]*self.scale)
                pnt2f = QPointF(self.pos.x(), self.pos.y())
                painter.drawLine(pnt1f, pnt2f)

        if self.show_ui:
            self.uir.draw(painter)
        # self.update_msg(painter)
        # self.visNN.show_results()
        painter.end()
        # self.paint_im.save('/tmp/screenshot.png')

    def update_msg(self, painter):
        # msgs = []
        if self.type is 'color':
            msg = 'coloring: (%d, %d, %d)' % (self.color.red(), self.color.green(), self.color.blue())
        if self.type is 'edge':
            if self.color is Qt.black:
                msg = 'sketching: add'
            else:
                msg = 'sketching: remove'
        if self.type is 'warp':
            msg = 'warping'
        # if self.type is 'patch':
            # msg = 'patch'
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
        # if self.type is 'patch':
            # self.brushWidth = self.uiPatch.update_width(d)

    def mousePressEvent(self, event):
        self.pos = self.round_point(event.pos())

        if event.button() == Qt.LeftButton:
            self.isPressed = True
            self.points.append(self.pos)
            self.update_opt_engine()
            self.update_ui()
            self.update()

        if event.button() == Qt.RightButton:
            if self.type is 'color':
                # QColorDialog.move((0,0))
                color = QColorDialog.getColor(parent=self)
                self.color = color

            if self.type is 'edge':
                if self.color is Qt.black:
                    self.color = Qt.white
                else:
                    self.color = Qt.black
                self.brushWidth = self.uiSketch.update_width(0, self.color)

            if self.type is 'warp':
                im = self.opt_engine.get_image(self.get_image_id(), self.get_frame_id())
                self.uiWarp.AddPoint(event.pos(), im)
                self.brushWidth = self.uiWarp.update_width(0)
            self.update()

    def mouseMoveEvent(self, event):
        # print('mouse move', self.pos)
        self.pos = self.round_point(event.pos())
        # print(self.pos)
        if self.isPressed:
            # point = event.pos()
            if self.type in ['color','edge']:
                self.points.append(self.pos)
            self.update_ui()
            self.update_opt_engine()
            self.update()
            # print(point)

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
        # frame_id_old = self.frame_id
        num_frames = self.opt_engine.get_num_frames()
        if num_frames > 0:
            self.frame_id = (self.frame_id+dif) % num_frames
            print("show frame id = %d"%self.frame_id)
            # self.update()

    def fix_z(self):
        self.opt_engine.init_z(self.get_frame_id(), self.get_image_id())
        # self.uiColor.reset()
        # self.uiSketch.reset()
        # self.uiWarp.reset()

    def morph_seq(self):
        print('generate morphing')
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
        self.color = QColor(255, 0, 0)
        self.brushWidth = self.uiColor.update_width(0)
        self.update()

    def use_edge(self):
        print('sketching')
        self.type = 'edge'
        self.color = Qt.black
        self.brushWidth = self.uiSketch.update_width(0, self.color)
        self.update()

    def use_warp(self):
        self.type = 'warp'
        self.color = Qt.black
        self.brushWidth = self.uiWarp.update_width(0)
        print('warp brush: %d' % self.brushWidth)
        self.update()


    def show_edits(self):
        self.show_ui = not self.show_ui
        self.update()

