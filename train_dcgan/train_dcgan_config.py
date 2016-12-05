def shoes_64():
    n_layers = 3 # number of layers
    n_f = 128  # number of feature channels
    npx = 64  # height = width
    nc = 3  # number of image channels
    nz = 100  # # of dim for Z
    niter = 25  # # of iter at starting learning rate
    niter_decay = 25  # of iter to linearly decay learning rate to zero
    return npx, n_layers, n_f, nc, nz, niter, niter_decay


def outdoor_64():
    n_layers = 3 # number of layers
    n_f = 128  # number of feature channels
    npx = 64  # height = width
    nc = 3  # number of image channels
    nz = 100  # # of dim for Z
    niter = 15  # # of iter at starting learning rate
    niter_decay = 15  # of iter to linearly decay learning rate to zero
    return npx, n_layers, n_f, nc, nz, niter, niter_decay

def church_64():
    n_layers = 3 # number of layers
    n_f = 128  # number of feature channels
    npx = 64  # height = width
    nc = 3  # number of image channels
    nz = 100  # # of dim for Z
    niter = 25  # # of iter at starting learning rate
    niter_decay = 25  # of iter to linearly decay learning rate to zero
    return npx, n_layers, n_f, nc, nz, niter, niter_decay


def handbag_64():
    n_layers = 3 # number of layers
    n_f = 128  # number of feature channels
    npx = 64  # height = width
    nc = 3  # number of image channels
    nz = 100  # # of dim for Z
    niter = 25  # # of iter at starting learning rate
    niter_decay = 25  # of iter to linearly decay learning rate to zero
    return npx, n_layers, n_f, nc, nz, niter, niter_decay

def hed_shoes_64():
    n_layers = 3 # number of layers
    n_f = 128  # number of feature channels
    npx = 64  # height = width
    nc = 1  # number of image channels
    nz = 100  # # of dim for Z
    niter = 25  # # of iter at starting learning rate
    niter_decay = 25  # of iter to linearly decay learning rate to zero
    return npx, n_layers, n_f, nc, nz, niter, niter_decay



def sketch_shoes_64():
    n_layers = 3 # number of layers
    n_f = 128  # number of feature channels
    npx = 64  # height = width
    nc = 1  # number of image channels
    nz = 100  # # of dim for Z
    niter = 25  # # of iter at starting learning rate
    niter_decay = 25  # of iter to linearly decay learning rate to zero
    return npx, n_layers, n_f, nc, nz, niter, niter_decay

def shoes_128():
    n_layers = 4 # number of layers
    n_f = 64  # number of feature channels
    npx = 128  # height = width
    nc = 3  # number of image channels
    nz = 100  # # of dim for Z
    niter = 25  # # of iter at starting learning rate
    niter_decay = 25  # of iter to linearly decay learning rate to zero
    return npx, n_layers, n_f, nc, nz, niter, niter_decay

