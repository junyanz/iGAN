from lib import html
import cv2
import os
import glob
import numpy as np
from pdb import set_trace as st


class ImageSave:
    def __init__(self, web_fold, title, append=False):
        self.web_fold = web_fold
        self.title = title
        self.html = html.HTML(web_fold, title)
        self.html.add_header(title)
        self.count = 0
        self.append = append

    def save_image(self, im_data, txts=None, header=None, width=400, cvt=False, imlinks=None, verbose=True):
        if txts is not None:
            # st()
            assert len(im_data) == len(txts)
        if imlinks is not None:
            assert len(imlinks) == len(im_data)

        if not self.append:
            self.html = html.HTML(self.web_fold, self.title)
            self.html.add_header(self.title)
        im_links = []
        if header is not None:
            self.html.add_header(header)

        img_fold = self.html.img_fold
        final_txts = []
        for n, im in enumerate(im_data):
            if im is not None:
                if imlinks is None:
                    im_link = 'row%4.4d_image%5.5d.png' % (self.count, n)
                else:
                    im_link = imlinks[n]
                im_links.append(im_link)
                if cvt:
                    if im.ndim == 3 and im.shape[2] == 3:
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    else:
                        im = np.squeeze(im)
                cv2.imwrite(os.path.join(img_fold, im_link), im)
                if verbose:
                    print('write image to %s' % os.path.join(img_fold, im_link))
                final_txts.append(txts[n])
        if txts is None:
            final_txts = im_links

        self.html.add_images(im_links, final_txts, im_links, width=width)
        self.count += 1

    def add_header(self, header):
        self.html.add_header(header)

    def save_folder(self, folder):
        # st()
        imgList = glob.glob('%s/*.{jpg,png,gif}' % folder)
        print('load %d images from %s' % (len(imgList), folder))
        print(imgList)
        ims = []
        txts = imgList
        for im_path in imgList:
            im = cv2.imread(os.path.join(folder, im_path))
            ims.append(im)
        self.save_image(ims, txts)

    def reset(self):
        self.count = 0
        self.html = html.HTML(self.web_fold, self.title)
        self.html.add_header(self.title)

    def save(self):
        self.html.save()
