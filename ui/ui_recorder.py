import copy
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import cv2

class UIRecorder:
    def __init__(self, shadow=False):
        self.strokes = []
        self.colors = []
        self.widths = []
        self.types = []
        self.patches = []
        self.shadow = shadow

    def save_record(self, stroke, color, width, type, patch=None):
        self.strokes.append(copy.deepcopy(stroke))
        self.colors.append(color)
        self.widths.append(width)
        self.types.append(type)
        self.patches.append(patch)

    def draw(self, painter):
        for points, color, width, t, patch in zip(self.strokes, self.colors, self.widths, self.types, self.patches):
            if t in ['edge', 'color']:
                if t is 'edge':
                    if self.shadow:
                        painter.setPen(QPen(color, 10, cap=Qt.RoundCap, join=Qt.RoundJoin))
                    else:
                        painter.setPen(QPen(Qt.gray, 10, Qt.DotLine, cap=Qt.RoundCap, join=Qt.RoundJoin))
                else:
                    painter.setPen(QPen(color, width, cap=Qt.RoundCap, join=Qt.RoundJoin))

                npnts = len(points)
                for i in range(0, npnts - 5, 5):
                    painter.drawLine(points[i], points[i + 5])
            if t is 'patch':
                w=patch.shape[0]
                h=patch.shape[1]
                qImg = QImage(patch.tostring(), w, h, QImage.Format_RGB888)
                painter.drawImage(points.x()-w/2, points.y()-h/2, qImg)


    def reset(self):
        del self.strokes[:]
        del self.colors[:]
        del self.widths[:]
        del self.types[:]
        del self.patches[:]