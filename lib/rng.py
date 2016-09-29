from numpy.random import RandomState
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from random import Random
from time import time
seed = int(time())
py_rng = Random(seed)
np_rng = RandomState(seed)
t_rng = RandomStreams(seed)


def set_seed(n):
    global seed, py_rng, np_rng, t_rng
    print('set seed = %d' % n)
    seed = n
    py_rng = Random(seed)
    np_rng = RandomState(seed)
    t_rng = RandomStreams(seed)
