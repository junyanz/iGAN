from PyQt4.QtCore import *
from PyQt4.QtGui import *
from . import gui_draw
from . import gui_vis
import time

class GUIDesign(QWidget):
    def __init__(self, opt_engine, win_size=320,
                 img_size=64, topK=16, model_name='tmp', useAverage=False, shadow=False):
        # draw the layout
        QWidget.__init__(self)
        morph_steps = 16
        self.opt_engine = opt_engine
        self.drawWidget = gui_draw.GUIDraw(opt_engine, win_size=win_size,
                                           img_size=img_size, topK=topK, useAverage=useAverage, shadow=shadow)
        self.drawWidget.setFixedSize(win_size, win_size)
        vbox = QVBoxLayout()

        self.drawWidgetBox = QGroupBox()
        self.drawWidgetBox.setTitle('Drawing Pad')
        vbox_t = QVBoxLayout()
        vbox_t.addWidget(self.drawWidget)
        self.drawWidgetBox.setLayout(vbox_t)
        vbox.addWidget(self.drawWidgetBox)
        self.slider = QSlider(Qt.Horizontal)
        vbox.addStretch(1)
        self.slider.setMinimum(0)
        self.slider.setMaximum(morph_steps-1)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)

        vbox.addWidget(self.slider)
        vbox.addStretch(1)
        self.bColor= QRadioButton("Coloring")
        self.bColor.setToolTip('The coloring brush allows the user to change the color of a specific region.')
        self.bEdge= QRadioButton("Sketching")
        self.bEdge.setToolTip('The sketching brush allows the user to outline the shape or add fine details.')
        self.bWarp = QRadioButton("Warping")
        self.bWarp.setToolTip('The warping brush allows the user to modify the shape more explicitly.')
        if shadow:
            self.bEdge.toggle()
            self.bColor.setDisabled(True)
            self.bWarp.setDisabled(True)
        else:
            self.bColor.toggle()
        bhbox =  QHBoxLayout()
        bGroup = QButtonGroup(self)
        bGroup.addButton(self.bColor)
        bGroup.addButton(self.bEdge)
        bGroup.addButton(self.bWarp)
        bhbox.addWidget(self.bColor)
        bhbox.addWidget(self.bEdge)
        bhbox.addWidget(self.bWarp)

        self.bEdit = QCheckBox('&Edits')
        self.bEdit.setChecked(True)
        # self.bAverage = QCheckBox('&AE')
        # self.bAverage.setChecked(useAverage)
        self.colorPush  = QPushButton()  # to visualize the selected color
        self.colorPush.setFixedWidth(20)
        self.colorPush.setFixedHeight(20)
        if shadow:
            self.colorPush.setStyleSheet("background-color: black")
        else:
            self.colorPush.setStyleSheet("background-color: green")
        bhbox.addWidget(self.colorPush)
        bhbox.addWidget(self.bEdit)
        # bhbox.addWidget(self.bAverage)


        vbox.addLayout(bhbox)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox)
        self.visWidgetBox = QGroupBox()
        self.visWidgetBox.setTitle('Candidate Results')
        vbox_t = QVBoxLayout()
        self.visWidget = gui_vis.GUI_VIS(opt_engine=opt_engine, grid_size=None, topK=topK, nps=win_size, model_name=model_name)
        vbox_t.addWidget(self.visWidget)
        self.visWidgetBox.setLayout(vbox_t)
        vbox2 = QVBoxLayout()

        vbox2.addWidget(self.visWidgetBox)
        vbox2.addStretch(1)

        self.bPlay = QPushButton('&Play')
        self.bPlay.setToolTip('Play a morphing sequence between the previous result and the current result')
        self.bFix = QPushButton('&Fix')
        self.bFix.setToolTip('Use the current result as a constraint')


        self.bRestart = QPushButton("&Restart")
        self.bRestart.setToolTip('Restart the system')
        self.bSave = QPushButton("&Save")
        self.bSave.setToolTip('Save the current result.')

        chbox = QHBoxLayout()
        chbox.addWidget(self.bPlay)
        chbox.addWidget(self.bFix)
        chbox.addWidget(self.bRestart)
        chbox.addWidget(self.bSave)


        vbox2.addLayout(chbox)

        hbox.addLayout(vbox2)
        self.setLayout(hbox)
        mainWidth = self.visWidget.winWidth + win_size + 60
        mainHeight = self.visWidget.winHeight + 100

        self.setGeometry(0, 0, mainWidth, mainHeight)
        self.setFixedSize(self.width(), self.height()) # fix window size
        # connect signals and slots
        self.connect(self.opt_engine, SIGNAL('update_image'), self.drawWidget.update_im)
        self.connect(self.opt_engine, SIGNAL('update_image'), self.visWidget.update_vis)
        self.connect(self.visWidget, SIGNAL('update_image_id'), self.drawWidget.set_image_id)
        self.connect(self.drawWidget, SIGNAL('update_image_id'), self.visWidget.set_image_id)
        self.slider.valueChanged.connect(self.visWidget.set_frame_id)
        self.slider.valueChanged.connect(self.drawWidget.set_frame_id)
        self.connect(self.drawWidget, SIGNAL('update_frame_id'), self.visWidget.set_frame_id)
        self.connect(self.drawWidget, SIGNAL('update_frame_id'), self.slider.setValue)
        self.opt_engine.start()
        self.drawWidget.update()
        self.visWidget.update()
        self.bColor.toggled.connect(self.drawWidget.use_color)
        self.bEdge.toggled.connect(self.drawWidget.use_edge)
        self.bWarp.toggled.connect(self.drawWidget.use_warp)
        self.colorPush.clicked.connect(self.drawWidget.change_color)
        self.connect(self.drawWidget, SIGNAL('update_color'), self.colorPush.setStyleSheet)
        self.bPlay.clicked.connect(self.play)
        self.bFix.clicked.connect(self.fix)
        self.bRestart.clicked.connect(self.reset)
        self.bSave.clicked.connect(self.save)
        self.bEdit.toggled.connect(self.show_edits)
        # self.bAverage.toggled.connect(self.drawWidget.change_average_mode)
        self.start_t = time.time()

    def reset(self):
        self.start_t = time.time()
        self.opt_engine.reset()
        self.drawWidget.reset()
        self.visWidget.reset()
        self.update()

    def play(self):
        self.drawWidget.morph_seq()

    def fix(self):
        self.drawWidget.fix_z()

    def show_edits(self):
        self.drawWidget.show_edits()

    def save(self):
        print('time spent = %3.3f' % (time.time()-self.start_t))
        self.visWidget.save()

    def change_average(self):
        self.drawWidget.change_average_mode()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
           self.reset()

        if event.key() == Qt.Key_A:
            self.change_average()

        if event.key() == Qt.Key_Q:
            print('time spent = %3.3f' % (time.time()-self.start_t))
            self.close()

        if event.key() == Qt.Key_F:
            self.fix()

        if event.key() == Qt.Key_E:
            self.drawWidget.show_edits()

        if event.key() == Qt.Key_P:
            self.play()

        if event.key() == Qt.Key_S:
            self.save()