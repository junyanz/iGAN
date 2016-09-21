import inspect, re
import numpy as np
from pprint import pprint
from pdb import set_trace as st
import os
import cv2
import cPickle

def debug_trace():
    '''Set a tracepoint in the Python debugger that works with Qt'''
    from PyQt4.QtCore import pyqtRemoveInputHook

    # Or for Qt5
    # from PyQt5.QtCore import pyqtRemoveInputHook

    from pdb import set_trace
    pyqtRemoveInputHook()
    set_trace()


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if callable(getattr(object, e))]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList])

def PickleLoad(file_name):
    with open(file_name, 'rb') as f:
        data= cPickle.load(f)
    return data

def PickleSave(file_name, data):
    with open(file_name, "wb") as f:
        cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def print_var(x):
    pprint(x)
    # name = varname(x)
    # pprint({name: x})


def interp_z(z0, z1, ratio, interp='linear'):
    print 'interp = %s' % interp
    if interp == 'linear':
        z_t = (1 - ratio) * z0 + ratio * z1

    if interp == 'slerp':
        N = len(z0)
        z_t = []
        for i in range(N):
            z0_i = z0[i]
            z1_i = z1[i]
            z0_n = z0_i / np.linalg.norm(z0_i)
            z1_n = z1_i / np.linalg.norm(z1_i)
            omega = np.arccos(np.dot(z0_n, z1_n))
            sin_omega = np.sin(omega)
            if sin_omega == 0:
                z_i = interp_z(z0_i, z1_i, ratio, 'linear')
            else:
                z_i = np.sin((1 - ratio) * omega) / sin_omega * z0_i + np.sin(ratio * omega) / sin_omega * z1_i
            z_t.append(z_i[np.newaxis,...])
        z_t = np.concatenate(z_t, axis=0)
    return z_t


def print_numpy(x, val=True, shp=False):
    # name = varname(x)
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def CVShow(im, im_name='', wait=1):
    print_numpy(im)
    # im_show = im
    if len(im.shape) >= 3 and im.shape[2] == 3:
        im_show = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        im_show = im

    cv2.imshow(im_name, im_show)
    cv2.waitKey(wait)
    return im_show


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, basestring):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)




def grayscale_grid_vis(X, (nh, nw)):
    # st()
    h, w = X[0].shape[:2]
    img = np.zeros((h * nh, w * nw))
    for n, x in enumerate(X):
        j = n / nw
        i = n % nw
        img[j * h:j * h + h, i * w:i * w + w] = x[:, :, 0]

    return img


def color_grid_vis(X, (nh, nw)):
    if X.shape[0] == 1:
        return X[0]

    h, w = X[0].shape[:2]
    if X.dtype == 'uint8':
        img = np.ones((h * nh, w * nw, 3), np.uint8) * 255
    else:
        img = np.ones((h * nh, w * nw, 3), X.dtype)

    for n, x in enumerate(X):
        j = n / nw
        i = n % nw
        img[j * h:j * h + h, i * w:i * w + w, :] = x
    return img

