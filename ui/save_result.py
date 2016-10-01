import os
from lib import image_save
from lib import utils
from PyQt4 import QtGui

class SaveResult():
    def __init__(self, model_name):
        self.isInit = False
        self.model_name = model_name
        self.default_dir = os.path.join('./web/', model_name)
        utils.mkdirs(self.default_dir)
        self.reset_count = 0
        self.save_count = 0

    def cache_result(self, ims=None,visIms=None, z=None):#, nn=None, z=None):
        self.ims = ims
        self.visIms = visIms
        self.z = z

    def save_seq(self, ims=None):
        if ims is not None:
            self.ims = ims

    def save(self):
        # if self.
        if not self.isInit:
            save_dir = QtGui.QFileDialog.getExistingDirectory(None,
                'Select a folder to save the result', self.default_dir,QtGui.QFileDialog.ShowDirsOnly)
            self.isInit = True
            self.save_dir = str(save_dir)
            utils.mkdirs(self.save_dir)
            self.html = image_save.ImageSave(self.save_dir, 'Gui screenshot', append=True)

        print('save the result to (%s)' % self.save_dir)

        if self.z is not None:
            self.z_dir = os.path.join(self.save_dir, 'z_vectors')
            utils.mkdirs(self.z_dir)
            utils.PickleSave(os.path.join(self.z_dir, 'z_drawing%3.3d_%3.3d' %(self.reset_count, self.save_count)), self.z)

        if self.ims is not None:
            txts=['']*self.ims.shape[0]
            self.html.save_image(self.ims, txts=txts, header='generated images (Drawing %3.3d, Step %3.3d)'  % (self.reset_count, self.save_count), cvt=True, width=128)
            self.html.save()
            self.save_count += 1

    def reset(self):
        self.gx = None
        self.nn = None
        self.z = None
        if self.isInit:
            self.html.add_header('New Drawing')
        self.reset_count += 1
        self.save_count = 0
