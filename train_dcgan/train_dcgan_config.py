def outdoor_64():
    n_layers = 3 # number of layers
    n_f = 128  # number of feature channels
    npx = 64  # height = width
    nc = 3  # number of image channels
    nz = 100  # # of dim for Z
    niter = 25  # # of iter at starting learning rate
    return npx, n_layers, n_f, nc, nz, niter


def shoes_64():
    n_layers = 3 # number of layers
    n_f = 128  # number of feature channels
    npx = 64  # height = width
    nc = 3  # number of image channels
    nz = 100  # # of dim for Z
    niter = 3  # # of iter at starting learning rate
    return npx, n_layers, n_f, nc, nz, niter

# def shoes_64():
#     n_layers = 3
#     n_f = 128
#     npx = 64
#     nc = 3
#     return npx, n_layers, n_f, nc
#
# def handbag_64():
#     n_layers = 3
#     n_f = 128
#     npx = 64
#     nc = 3
#     return npx, n_layers, n_f, nc
#
# def church_64():
#     n_layers = 3
#     n_f = 128
#     npx = 64
#     nc = 3
#     return npx, n_layers, n_f, nc
#
