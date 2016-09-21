import theano
import theano.tensor as T
from lib.theano_utils import floatX, sharedX


class Softmax(object):

    def __init__(self):
        pass

    def __call__(self, x):
        e_x = T.exp(x - x.max(axis=1).dimshuffle(0, 'x'))
        return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

class ConvSoftmax(object):

    def __init__(self):
        pass

    def __call__(self, x):
        e_x = T.exp(x - x.max(axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

class Maxout(object):

    def __init__(self, n_pool=2):
        self.n_pool = n_pool

    def __call__(self, x):
        if x.ndim == 2:
            x = T.max([x[:, n::self.n_pool] for n in range(self.n_pool)], axis=0)
        elif x.ndim == 4:
            x = T.max([x[:, n::self.n_pool, :, :] for n in range(self.n_pool)], axis=0)
        else:
            raise NotImplementedError
        return x

class Rectify(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return (x + T.abs_(x)) / 2.0

class ClippedRectify(object):

    def __init__(self, clip=10.):
        self.clip = clip

    def __call__(self, x):
        return T.clip((x + T.abs_(x)) / 2.0, 0., self.clip)

class LeakyRectify(object):

    def __init__(self, leak=0.2):
        self.leak = sharedX(leak)

    def __call__(self, x):
        f1 = 0.5 * (1 + self.leak)
        f2 = 0.5 * (1 - self.leak)
        return f1 * x + f2 * T.abs_(x)

class Prelu(object):

    def __init__(self):
        pass

    def __call__(self, x, leak):
        if x.ndim == 4:
            leak = leak.dimshuffle('x', 0, 'x', 'x')
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * T.abs_(x)

class Tanh(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return T.tanh(x)

class Sigmoid(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return T.nnet.sigmoid(x)

class Linear(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return x

class HardSigmoid(object):

    def __init__(self):
        pass

    def __call__(self, X):
        return T.clip(X + 0.5, 0., 1.)

class TRec(object):

    def __init__(self, t=1):
        self.t = t

    def __call__(self, X):
        return X*(X > self.t)

class HardTanh(object):

    def __init__(self):
        pass

    def __call__(self, X):
        return T.clip(X, -1., 1.)