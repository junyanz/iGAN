from __future__ import print_function
import sys
sys.path.append('..')
import os
import json
from time import time
import numpy as np
from tqdm import tqdm
import theano
import theano.tensor as T
import train_dcgan_config
from lib import updates
from lib import utils
from lib.rng import py_rng, np_rng
from lib import costs
import train_dcgan_utils
from lib.theano_utils import floatX, sharedX
import load
from lib import image_save
import argparse

# set arguments and parameters
parser = argparse.ArgumentParser('Train DCGAN model')
parser.add_argument('--model_name', dest='model_name', help='model name', default='shoes_64', type=str)
parser.add_argument('--ext', dest='ext', help='experiment name=model_name+ext', default='', type=str)
parser.add_argument('--data_file', dest='data_file', help='the file that stores the hdf5 data', type=str, default=None)
parser.add_argument('--cache_dir', dest='cache_dir', help='cache directory that stores models, samples and webpages', type=str, default=None)
parser.add_argument('--batch_size', dest='batch_size', help='the number of examples in each batch', type=int, default=128)

parser.add_argument('--update_k', dest='update_k', help='the number of discrim updates for each gen update', type=int, default=2)
parser.add_argument('--save_freq', dest='save_freq', help='save a model every save_freq epochs', type=int, default=1)
parser.add_argument('--lr', dest='lr', help='learning rate', type=float, default=0.0002)
parser.add_argument('--weight_decay', dest='weight_decay', help='l2 weight decay', type=float, default=1e-5)
parser.add_argument('--b1', dest='b1', help='momentum term of adam', type=float, default=0.5)
args = parser.parse_args()


if not args.data_file:
    args.data_file = '../datasets/%s.hdf5' % args.model_name

n_vis = 196
npx, n_layers, n_f, nc, nz, niter, niter_decay = getattr(train_dcgan_config, args.model_name)()
expr_name = args.model_name + args.ext

if not args.cache_dir:
    args.cache_dir = './cache/%s/' % expr_name

for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

# create directories
sample_dir = os.path.join(args.cache_dir, 'samples')
model_dir = os.path.join(args.cache_dir, 'models')
log_dir = os.path.join(args.cache_dir, 'log')
web_dir = os.path.join(args.cache_dir, 'web_dcgan')
html = image_save.ImageSave(web_dir, expr_name, append=True)
utils.mkdirs([sample_dir, model_dir, log_dir, web_dir])

# load data from hdf5 file
tr_data, te_data, tr_stream, te_stream, ntrain, ntest = load.load_imgs(ntrain=None, ntest=None, batch_size=args.batch_size,data_file=args.data_file)
te_handle = te_data.open()
test_x, = te_data.get_data(te_handle, slice(0, ntest))

# generate real samples and test transform/inverse_transform
test_x = train_dcgan_utils.transform(test_x, nc=nc)
vis_idxs = py_rng.sample(np.arange(len(test_x)), n_vis)
vaX_vis = train_dcgan_utils.inverse_transform(test_x[vis_idxs], npx=npx, nc=nc)
# st()
n_grid = int(np.sqrt(n_vis))
grid_real = utils.grid_vis((vaX_vis*255.0).astype(np.uint8), n_grid, n_grid)
train_dcgan_utils.save_image(grid_real, os.path.join(sample_dir, 'real_samples.png'))


# define DCGAN model
disc_params = train_dcgan_utils.init_disc_params(n_f=n_f, n_layers=n_layers, nc=nc)
gen_params = train_dcgan_utils.init_gen_params(nz=nz, n_f=n_f, n_layers=n_layers, nc=nc)
x = T.tensor4()
z = T.matrix()

gx = train_dcgan_utils.gen(z, gen_params, n_layers=n_layers, n_f=n_f, nc=nc)
p_real = train_dcgan_utils.discrim(x, disc_params, n_layers=n_layers)
p_gen = train_dcgan_utils.discrim(gx, disc_params, n_layers=n_layers)

d_cost_real = costs.bce(p_real, T.ones(p_real.shape))
d_cost_gen = costs.bce(p_gen, T.zeros(p_gen.shape))
g_cost_d = costs.bce(p_gen, T.ones(p_gen.shape))

d_cost = d_cost_real + d_cost_gen
g_cost = g_cost_d

cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

lrt = sharedX(args.lr)
d_updater = updates.Adam(lr=lrt, b1=args.b1, regularizer=updates.Regularizer(l2=args.weight_decay))
g_updater = updates.Adam(lr=lrt, b1=args.b1, regularizer=updates.Regularizer(l2=args.weight_decay))
d_updates = d_updater(disc_params, d_cost)
g_updates = g_updater(gen_params, g_cost)
updates = d_updates + g_updates

print('COMPILING')
t = time()
_train_g = theano.function([x, z], cost, updates=g_updates)
_train_d = theano.function([x, z], cost, updates=d_updates)
_gen = theano.function([z], gx)
print('%.2f seconds to compile theano functions' % (time() - t))

# test z samples
sample_zmb = floatX(np_rng.uniform(-1., 1., size=(n_vis, nz)))


f_log = open('%s/training_log.ndjson' % log_dir, 'wb')
log_fields = ['n_epochs', 'n_updates', 'n_examples', 'n_seconds', 'g_cost', 'd_cost',]

# initialization
n_updates = 0
n_epochs = 0
n_examples = 0
t = time()

for epoch in range(niter+niter_decay):
    for imb, in tqdm(tr_stream.get_epoch_iterator(), total=ntrain / args.batch_size):
        imb = train_dcgan_utils.transform(imb, nc=nc)
        zmb = floatX(np_rng.uniform(-1., 1., size=(len(imb), nz)))
        if n_updates % args.update_k == 0:
            cost = _train_g(imb, zmb)
        else:
            cost = _train_d(imb, zmb)
        n_updates += 1
        n_examples += len(imb)

    g_cost = float(cost[0])
    d_cost = float(cost[1])
    # print logging information
    log = [n_epochs, n_updates, n_examples, time() - t,  g_cost, d_cost]
    print('epoch %.0f: G_cost %.4f, D_cost %.4f' % (epoch,  g_cost, d_cost))
    f_log.write(json.dumps(dict(zip(log_fields, log))) + '\n')
    f_log.flush()

    n_epochs += 1

    # generate samples and write webpage
    samples = np.asarray(_gen(sample_zmb))
    samples_t = train_dcgan_utils.inverse_transform(samples, npx=npx, nc=nc)
    grid_vis = utils.grid_vis(samples_t, n_grid, n_grid)
    grid_vis_i = (grid_vis*255.0).astype(np.uint8)
    train_dcgan_utils.save_image(grid_vis_i, os.path.join(sample_dir, 'gen_%5.5d.png'%n_epochs))
    html.save_image([grid_vis_i], [''], header='epoch_%3.3d' % n_epochs, width=grid_vis.shape[1], cvt=True)
    html.save()

    # save models
    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - args.lr / niter_decay))
    if n_epochs % args.save_freq == 0:
        train_dcgan_utils.save_model(disc_params, '%s/disc_params_%3.3d' % (model_dir, n_epochs))
        train_dcgan_utils.save_model(gen_params, '%s/gen_params_%3.3d' % (model_dir, n_epochs))
    train_dcgan_utils.save_model(disc_params, '%s/disc_params' % model_dir)
    train_dcgan_utils.save_model(gen_params, '%s/gen_params' % model_dir)
