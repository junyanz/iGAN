from __future__ import print_function
import sys
sys.path.append('..')
import numpy as np
from tqdm import tqdm
import os
import theano
import theano.tensor as T
import train_dcgan_utils
import train_dcgan_config
from lib import utils
from lib.rng import np_rng
from lib.theano_utils import floatX
import argparse
from time import time

# set parameters and arguments
parser = argparse.ArgumentParser('compute batchnorm statistics for DCGAN model')
parser.add_argument('--model_name', dest='model_name', help='model name', default='shoes_64', type=str)
parser.add_argument('--ext', dest='ext', help='experiment name=model_name+ext', default='', type=str)
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

for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

model_dir = os.path.join(args.cache_dir, 'models')
disc_bn_path = os.path.join(model_dir, 'disc_batchnorm')
gen_bn_path = os.path.join(model_dir, 'gen_batchnorm')

# load DCGAN model
disc_params = train_dcgan_utils.init_disc_params(n_f=n_f, n_layers=n_layers, nc=nc)
gen_params = train_dcgan_utils.init_gen_params(nz=nz, n_f=n_f, n_layers=n_layers, nc=nc)

print('load model from %s, expr_name=%s' % (model_dir, expr_name))
train_dcgan_utils.load_model(disc_params, os.path.join(model_dir, 'disc_params'))
train_dcgan_utils.load_model(gen_params, os.path.join(model_dir, 'gen_params'))

Z = T.matrix()
gX, gbn = train_dcgan_utils.gen_batchnorm(Z, gen_params, n_layers=n_layers, n_f=n_f, nc=nc)
p_gen, dbn = train_dcgan_utils.discrim_batchnorm(gX, disc_params, n_layers=n_layers)
ngbn = len(gbn)
ndbn = len(dbn)
bn_data = gbn + dbn

print('COMPILING...')
t = time()
_estimate_bn = theano.function([Z], bn_data)
print('%.2f seconds to compile theano functions' % (time() - t))


# batchnorm statistics
nb_sum = []
nb_mean = []
nb_mean_ext = []


# first pass
print('first pass: computing mean')
for n in tqdm(range(num_batches)):
    zmb = floatX(np_rng.uniform(-1., 1., size=(batch_size, nz)))
    bn_data = _estimate_bn(zmb)

    if n == 0:
        for d in bn_data:
            nb_sum.append(d)
    else:
        for id, d in enumerate(bn_data):
            nb_sum[id] = nb_sum[id] + d

# compute empirical mean
for id, d_sum in enumerate(nb_sum):
    if d_sum.ndim == 4:
        m = np.mean(d_sum, axis=(0, 2, 3)) / num_batches
        nb_mean.append(m)
        nb_mean_ext.append(np.reshape(m, [1, len(m), 1, 1]))
    if d_sum.ndim == 2:
        m = np.mean(d_sum, axis=0) / float(num_batches)
        nb_mean.append(m)
        nb_mean_ext.append(m)


# second pass
nb_var_sum = []
print('second pass: computing variance')
for n in tqdm(range(num_batches)):
    zmb = floatX(np_rng.uniform(-1., 1., size=(batch_size, nz)))
    bn_data = _estimate_bn(zmb)
    if n == 0:
        for id, d in enumerate(bn_data):
            var = (np.copy(d) - nb_mean_ext[id]) ** 2
            nb_var_sum.append(var)
    else:
        for id, d in enumerate(bn_data):
            var = (np.copy(d) - nb_mean_ext[id]) ** 2
            nb_var_sum[id] = nb_var_sum[id] + var

# compute empirical variance
nb_var = []
for id, var_sum in enumerate(nb_var_sum):
    if var_sum.ndim == 4:
        nb_var.append(np.mean(var_sum, axis=(0, 2, 3)) / float(num_batches - 1))

    if var_sum.ndim == 2:
        nb_var.append(np.mean(var_sum, axis=0) / float(num_batches - 1))

# save batchnorm mean and var for disc and gen
gen_batchnorm = nb_mean[:ngbn] + nb_var[:ngbn]
disc_batchnorm = nb_mean[ngbn:] + nb_var[ngbn:]
utils.PickleSave(gen_bn_path, gen_batchnorm)
utils.PickleSave(disc_bn_path, disc_batchnorm)

