import numpy as np
import theano

def intX(X):
    return np.asarray(X, dtype=np.int32)

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

def shared0s(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)

def sharedNs(shape, n, dtype=theano.config.floatX, name=None):
    return sharedX(np.ones(shape)*n, dtype=dtype, name=name)