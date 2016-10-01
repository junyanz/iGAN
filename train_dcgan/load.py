import os
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from time import time


def load_imgs(ntrain=None, ntest=None, batch_size=128, data_file=None):
    t = time()
    print('LOADING DATASET...')
    path = os.path.join(data_file)
    tr_data = H5PYDataset(path, which_sets=('train',))
    te_data = H5PYDataset(path, which_sets=('test',))

    if ntrain is None:
        ntrain = tr_data.num_examples
    else:
        ntrain = min(ntrain, tr_data.num_examples)

    if ntest is None:
        ntest = te_data.num_examples
    else:
        ntest = min(ntest, te_data.num_examples)
    print('name = %s, ntrain = %d, ntest = %d' % (data_file, ntrain, ntest))

    tr_scheme = ShuffledScheme(examples=ntrain, batch_size=batch_size)
    tr_stream = DataStream(tr_data, iteration_scheme=tr_scheme)

    te_scheme = ShuffledScheme(examples=ntest, batch_size=batch_size)
    te_stream = DataStream(te_data, iteration_scheme=te_scheme)
    print('%.2f secs to load data' % (time() - t))
    return tr_data, te_data, tr_stream, te_stream, ntrain, ntest


def load_imgs_seq(ntrain=None, ntest=None, batch_size=128, data_file=None):
    t = time()
    print('LOADING DATASET...')
    path = os.path.join(data_file)
    tr_data = H5PYDataset(path, which_sets=('train',))
    te_data = H5PYDataset(path, which_sets=('test',))

    if ntrain is None:
        ntrain = tr_data.num_examples
    if ntest is None:
        ntest = te_data.num_examples

    tr_scheme = SequentialScheme(examples=ntrain, batch_size=batch_size)
    tr_stream = DataStream(tr_data, iteration_scheme=tr_scheme)

    te_scheme = SequentialScheme(examples=ntest, batch_size=batch_size)
    te_stream = DataStream(te_data, iteration_scheme=te_scheme)

    print('name = %s, ntrain = %d, ntest = %d' % (data_file, ntrain, ntest))
    print('%.2f seconds to load data' % (time() - t))

    return tr_data, te_data, tr_stream, te_stream, ntrain, ntest


def load_imgs_raw(ntrain=None, ntest=None, data_file=None):
    t = time()
    print('LOADING DATASET...')
    path = os.path.join(data_file)
    tr_data = H5PYDataset(path, which_sets=('train',))
    te_data = H5PYDataset(path, which_sets=('test',))

    if ntrain is None:
        ntrain = tr_data.num_examples
    if ntest is None:
        ntest = te_data.num_examples

    print('name = %s, ntrain = %d, ntest = %d' % (data_file, ntrain, ntest))
    print('%.2f seconds to load data' % (time() - t))

    return tr_data, te_data, ntrain, ntest
