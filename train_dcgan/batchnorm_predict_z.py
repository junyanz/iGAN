from __future__ import print_function
import sys
sys.path.append('..')
import numpy as np
from tqdm import tqdm
import theano
import theano.tensor as T
from load import load_imgs
import argparse
import train_dcgan_utils
import train_dcgan_config
import os
from time import time
from lib import utils

# set parameters and arguments
parser = argparse.ArgumentParser('compute batchnorm statistics for DCGAN model')
parser.add_argument('--model_name', dest='model_name', help='model name', default='shoes_64', type=str)
parser.add_argument('--ext', dest='ext', help='experiment name=model_name+ext', default='', type=str)
parser.add_argument('--data_file', dest='data_file', help='the file that stores the hdf5 data', type=str, default=None)
parser.add_argument('--batch_size', dest='batch_size', help='the number of examples in each batch', type=int, default=128)
parser.add_argument('--num_batches', dest='num_batches', help='number of batches for estimating batchnorm parameters',type=int, default=1000)
parser.add_argument('--cache_dir', dest='cache_dir', help='cache directory that stores models, samples and webpages', type=str, default=None)
args = parser.parse_args()

expr_name = args.model_name + args.ext
npx, n_layers, n_f, nc, nz, niter, niter_decay = getattr(train_dcgan_config, args.model_name)()
num_batches = args.num_batches
batch_size = args.batch_size

if not args.cache_dir:
    args.cache_dir = './cache/%s/' % expr_name

if not args.data_file:
    args.data_file = '../datasets/%s.hdf5' % args.model_name


for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

model_dir = os.path.join(args.cache_dir, 'models')
predict_bn_path = os.path.join(model_dir, 'predict_batchnorm')
ntrain = batch_size * num_batches

# load data
tr_data, te_data, tr_stream, te_stream, ntrain, ntest = load_imgs(ntrain=ntrain, ntest=None, batch_size=batch_size, data_file=args.data_file)

# define theano model
predict_params = train_dcgan_utils.init_predict_params(nz=nz, n_f=n_f, n_layers=n_layers)
print('load predictive model from %s' % model_dir)
train_dcgan_utils.load_model(predict_params, os.path.join(model_dir, 'predict_params'))

x = T.tensor4()
z_p, bn_data = train_dcgan_utils.predict_batchnorm(x, predict_params, n_layers=n_layers)

print('COMPILING...')
t = time()
_estimate_batchnorm = theano.function([x], bn_data)
print('%.2f seconds to compile theano functions' % (time() - t))


nb_sum = []
nb_mean = []
nb_mean_ext = []
num_batches = int(np.floor(ntrain / float(batch_size)))
print('n_batch = %d, batch_size = %d' % (num_batches, batch_size))

# first pass
print('first pass: computing mean')
n = 0
for imb, in tqdm(tr_stream.get_epoch_iterator(), total=ntrain / batch_size):
    imb = train_dcgan_utils.transform(imb, nc=nc)
    bn_data = _estimate_batchnorm(imb)

    if n == 0:
        for d in bn_data:
            nb_sum.append(d)
    else:
        for id, d in enumerate(bn_data):
            nb_sum[id] = nb_sum[id] + d
    n = n+1
    if n >= num_batches:
        break
# compute empirical mean
for id, d_sum in enumerate(nb_sum):
    if d_sum.ndim == 4:
        m = np.mean(d_sum, axis=(0, 2, 3)) / num_batches
        nb_mean.append(m)
        nb_mean_ext.append(np.reshape(m, [1, len(m), 1, 1]))
    if d_sum.ndim == 2:
        m = np.mean(d_sum, axis=0) / num_batches
        nb_mean.append(m)
        nb_mean_ext.append(m)


# second pass
print('second pass: computing variance')
tr_stream.reset()
nb_var_sum = []
n = 0

for imb, in tqdm(tr_stream.get_epoch_iterator(), total=ntrain / batch_size):
    imb = train_dcgan_utils.transform(imb, nc=nc)
    bn_data = _estimate_batchnorm(imb)
    if n == 0:
        for id, d in enumerate(bn_data):
            var = (np.copy(d) - nb_mean_ext[id]) ** 2
            nb_var_sum.append(var)
    else:
        for id, d in enumerate(bn_data):
            var = (np.copy(d) - nb_mean_ext[id]) ** 2
            nb_var_sum[id] = nb_var_sum[id] + var
    n += 1
    if n >= num_batches:
        break

# compute empirical variance
nb_var = []
for id, var_sum in enumerate(nb_var_sum):
    if var_sum.ndim == 4:
        nb_var.append(np.mean(var_sum, axis=(0, 2, 3)) / (num_batches - 1))

    if var_sum.ndim == 2:
        nb_var.append(np.mean(var_sum, axis=0) / (num_batches - 1))

print('saving model to %s' % predict_bn_path)
predict_batchnorm = nb_mean + nb_var
utils.PickleSave(predict_bn_path, predict_batchnorm)
