
def resize(im, shape):
    try:
        import cv2
        return cv2.resize(im, shape)
    except ImportError:
        pass
    from skimage import transform
    import numpy as np
    return (transform.resize(im.astype(np.float32)/255., list(shape)+[3])*255).astype(im.dtype)

def line(im, p1, p2, c, t=1):
    try:
        import cv2
        return cv2.line(im, p1, p2, c, t)
    except ImportError:
        pass
    from skimage import draw
    import numpy as np
    if t!=1:
        assert "Thickness t=%d not yet supported"%t
    x,y = draw.line(int(p1[0]),int(p1[1]),int(p2[0]),int(p2[1]))
    valid = np.logical_and.reduce((0<=x, x<im.shape[1], 0<=y, y<im.shape[0]))
    im[y[valid],x[valid]] = c

def imwrite(path, im):
    try:
        import cv2
        return cv2.imwrite(path, im)
    except ImportError:
        pass
    from skimage import io
    return io.imwrite(path, im)


def imread(path):
    try:
        import cv2
        return cv2.imread(path)
    except ImportError:
        pass
    from skimage import io
    return io.imread(path)

def min_resize(x, size):
    """
    Resize an image so that it is size along the minimum spatial dimension.
    """
    w, h = list(map(float, x.shape[:2]))
    if min([w, h]) != size:
        if w <= h:
            x = resize(x, (int(round((h/w)*size)), int(size)), interpolation=interpolation)
        else:
            x = resize(x, (int(size), int(round((w/h)*size))), interpolation=interpolation)
    return x
