from __future__ import print_function
import sys
sys.path.append('..')
import h5py
import os
import numpy as np
from fuel.datasets.hdf5 import H5PYDataset
import cv2
import argparse


def print_name(name):
    print(name)

# set arguments and parameters
parser = argparse.ArgumentParser('create a hdf5 file from lmdb or image directory')
parser.add_argument('--dataset_dir', dest='dataset_dir', help='the file or directory that stores the image collection', type=str)
parser.add_argument('--width', dest='width', help='image size: width x width', type=int, default=64)
parser.add_argument('--mode', dest='mode', help='how the image collection is stored (mnist, lmdb, dir)', type=str, default='dir')
parser.add_argument('--channel', dest='channel', help='the number of image channels', type=int, default='3')
parser.add_argument('--hdf5_file', dest='hdf5_file', help='output hdf5 file', type=str)
args = parser.parse_args()


for arg in vars(args):
    print('[%s] = ' % arg,  getattr(args, arg))

width = args.width

# process the image
def ProcessImage(img, channel=3):  # [assumption]:  image is x, w, 3 with uint8
    if channel == 1:
        img = 255- cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(1, width, width, 1)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(1, width, width, 3)
    return img

if os.path.isfile(args.hdf5_file): # if hdf5 file already exists
    print('already created  %s' % args.hdf5_file)
    f = h5py.File(args.hdf5_file, 'r')
    f.visit(print_name)
else:
    if args.mode == 'mnist':  # read mnist dataset
        fd = open(args.dataset_dir)
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        imgs = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.uint8)
        nImgs = imgs.shape[0]
        np.random.shuffle(imgs)

    if args.mode == 'dir':  # if images are stored in a directory
        imgList = os.listdir(args.dataset_dir)
        nImgs = len(imgList)
        np.random.shuffle(imgList)
        print('read %d images from %s' % (nImgs, args.dataset_dir))

        imgs = []
        for id, file in enumerate(imgList):
            if id % 1000 == 0:
                print('read %d/%d image' % (id, nImgs))
            img = cv2.imread(os.path.join(args.dataset_dir, file), cv2.IMREAD_COLOR)
            img = cv2.resize(img, dsize=(width, width), interpolation=cv2.INTER_CUBIC)
            img = ProcessImage(img, args.channel)
            imgs.append(img)
        nImgs = len(imgs)
        imgs = np.concatenate(imgs, axis=0)


    if args.mode == 'lmdb':  # convert lmdb to hdf5
        print('you need to install lmdb')
        print('sudo pip install lmdb')
        import lmdb
        env = lmdb.open(args.dataset_dir, map_size=1099511627776,
                    max_readers=100, readonly=True)
        imgs = list()
        id = 0
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, val in cursor:
                id = id + 1
                if id % 1000 == 0:
                    print('read %d images' % id)
                img = cv2.imdecode(np.fromstring(val, dtype=np.uint8),flags=1)
                img = cv2.resize(img, dsize=(width, width),interpolation = cv2.INTER_CUBIC)
                img = ProcessImage(imgs, args.channel)
                imgs.append(img)
            nImgs = len(imgs)
            imgs = np.concatenate(imgs, axis=0)
            imgIdx = np.arange(nImgs)
            np.random.shuffle(imgIdx)
            imgs = imgs[imgIdx]


    f = h5py.File(args.hdf5_file, 'w')
    print('writing %d images to hdf5 file %s' % (nImgs, args.hdf5_file))
    f_imgs = f.create_dataset('imgs', data=imgs)
    f_imgs.dims[0].label = 'batch'
    f_imgs.dims[1].label = 'height'
    f_imgs.dims[2].label = 'width'
    f_imgs.dims[3].label = 'channel'

    nVal = min(int(nImgs*0.05), 10000)
    split_dict = { 'train': {'imgs': (0, nImgs-nVal)},
                   'test': {'imgs': (nImgs-nVal, nImgs)}}
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()
