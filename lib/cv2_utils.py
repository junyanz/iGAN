import cv2

def min_resize(x, size, interpolation=cv2.INTER_LINEAR):
    """
    Resize an image so that it is size along the minimum spatial dimension.
    """
    w, h = map(float, x.shape[:2])
    if min([w, h]) != size:
        if w <= h:
            x = cv2.resize(x, (int(round((h/w)*size)), int(size)), interpolation=interpolation)
        else:
            x = cv2.resize(x, (int(size), int(round((w/h)*size))), interpolation=interpolation)
    return x