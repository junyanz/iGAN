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
from model_def import dcgan_theano
from lib.theano_utils import floatX, sharedX
import load
from lib import image_save
import argparse
import cv2

parser = argparse.ArgumentParser('--desc')
parser.add_argument('--class_name', dest='class_name', help='category name', default='outdoor_64', type=str)
parser.add_argument('--ext', dest='ext', help='ext for the experiment name', default='', type=str)
parser.add_argument('--data_file', dest='data_file', help='the file that stores the hdf5 data', type=str,
                    default=None)

parser.add_argument('--batch_size', dest='batch_size', help='the number of examples in each batch', type=int,
                    default=128)

parser.add_argument('--update_k', dest='top_k', help='the number of discrim updates for each gen update', type=int,
                    default=1)
parser.add_argument('--lr', dest='lr', help='learning rate', type=float, default=0.0002)
parser.add_argument('--weight_decay', dest='weight_decay', help='l2 weight decay', type=float, default=1e-5)
parser.add_argument('--b1', dest='b1', help='momentum term of adam', type=float, default=0.5)
args = parser.parse_args()
if not args.data_file:
    args.data_file = '../datasets/%s.hdf5' % args.class_name


for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

n_vis = 196
npx, n_layers, n_f, nc, nz, niter= getattr(train_dcgan_config, args.class_name)()
niter_decay = niter  # of iter to linearly decay learning rate to zero
expr_name = args.class_name + args.ext

cache_dir = '../cache/%s/' % expr_name
sample_dir = os.path.join(cache_dir, 'samples')
model_dir = os.path.join(cache_dir, 'models')
log_dir = os.path.join(cache_dir, 'log')
web_dir = os.path.join(cache_dir, 'web')
html = image_save.ImageSave(web_dir, expr_name, append=True)

utils.mkdirs([sample_dir, model_dir, log_dir, web_dir])


tr_data, te_data, tr_stream, te_stream, ntrain, ntest = load.load_imgs(ntrain=None, ntest=None, batch_size=args.batch_size,data_file=args.data_file)
te_handle = te_data.open()
test_x, = te_data.get_data(te_handle, slice(0, ntest))
test_x = dcgan_theano.transform(test_x)
vis_idxs = py_rng.sample(np.arange(len(test_x)), n_vis)
vaX_vis = dcgan_theano.inverse_transform(test_x[vis_idxs], npx=npx)
# st()
n_grid = int(np.sqrt(n_vis))
utils.grid_vis(vaX_vis, (n_grid, n_grid),  os.path.join(sample_dir,'real_samples.png'))
exit(-1)

disc_params = dcgan_theano.init_disc_params(nz=nz, n_f=n_f, n_layers=n_layers, nc=nc)
gen_params = dcgan_theano.init_gen_params(nz=nz, n_f=n_f, n_layers=n_layers, nc=nc)
#
# if c_iter > 0:
#     print('load model from %s' % model_dir)
#     modeldef.load_model(disc_params, '%s/disc_params_%3.3d' % (model_dir, c_iter))
#     modeldef.load_model(gen_params, '%s/gen_params_%3.3d' % (model_dir, c_iter))

x = T.tensor4()
z = T.matrix()

gx = dcgan_theano.gen(z, gen_params, n_layers=n_layers, n_f=n_f, nc=nc)
p_real = dcgan_theano.discrim(x, disc_params, n_layers=n_layers, nc=nc)
p_gen = dcgan_theano.discrim(gx, disc_params, n_layers=n_layers, nc=nc)

d_cost_real = costs.bce(p_real, T.ones(p_real.shape))#.mean()
d_cost_gen = costs.bce(p_gen, T.zeros(p_gen.shape))#.mean()
g_cost_d = costs.bce(p_gen, T.ones(p_gen.shape))#.mean()

d_cost = d_cost_real + d_cost_gen
g_cost = g_cost_d

cost = [g_cost, d_cost, g_cost_d, d_cost_real, d_cost_gen]

lrt = sharedX(args.lr)
d_updater = updates.Adam(lr=lrt, b1=args.b1, regularizer=updates.Regularizer(l2=args.weight_decay))
g_updater = updates.Adam(lr=lrt, b1=args.b1, regularizer=updates.Regularizer(l2=args.weight_decay))
d_updates = d_updater(disc_params, d_cost)
g_updates = g_updater(gen_params, g_cost)
updates = d_updates + g_updates

print 'COMPILING'
t = time()
_train_g = theano.function([x, z], cost, updates=g_updates)
_train_d = theano.function([x, z], cost, updates=d_updates)
_gen = theano.function([z], gx)
print '%.2f seconds to compile theano functions' % (time() - t)

sample_zmb = floatX(np_rng.uniform(-1., 1., size=(n_vis, nz)))


def gen_samples(n, nbatch=128):
    g_imgs = []
    n_gen = 0
    for i in range(n / nbatch):
        zmb = floatX(np_rng.uniform(-1., 1., size=(nbatch, nz)))
        xmb = _gen(zmb)
        g_imgs.append(xmb)
        n_gen += len(xmb)
    n_left = n - n_gen
    zmb = floatX(np_rng.uniform(-1., 1., size=(n_left, nz)))
    xmb = _gen(zmb)
    g_imgs.append(xmb)
    return np.concatenate(g_imgs, axis=0)


f_log = open('%s/training_log.ndjson' % log_dir, 'wb')
log_fields = [
    'n_epochs',
    'n_updates',
    'n_examples',
    'n_seconds',
    'g_cost',
    'd_cost',
]

test_x = test_x.reshape(len(test_x), -1)


n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()
for epoch in range(niter+niter_decay):

    for imb, in tqdm(tr_stream.get_epoch_iterator(), total=ntrain / args.batch_size):
        imb = dcgan_theano.transform(imb)
        zmb = floatX(np_rng.uniform(-1., 1., size=(len(imb), nz)))
        if n_updates % (args.update_k + 1) == 0:
            cost = _train_g(imb, zmb)
        else:
            cost = _train_d(imb, zmb)
        n_updates += 1
        n_examples += len(imb)
    g_cost = float(cost[0])
    d_cost = float(cost[1])
    gX = gen_samples(10000)
    gX = gX.reshape(len(gX), -1)

    log = [n_epochs, n_updates, n_examples, time() - t,  g_cost, d_cost]  # va_nnd_10k, va_nnd_100k,
    print('epoch %.0f: G_cost %.4f, D_cost %.4f' % (epoch,  g_cost, d_cost))  # va_nnd_10k, va_nnd_100k,
    f_log.write(json.dumps(dict(zip(log_fields, log))) + '\n')
    f_log.flush()

    samples = np.asarray(_gen(sample_zmb))
    grid_vis = utils.grid_vis(dcgan_theano.inverse_transform(samples, npx=npx), n_grid, n_grid)#, '%s/gen_%5.5d.png' % (sample_dir, n_epochs))
    grid_vis_i = (grid_vis*255).astype(np.uint8)
    grid_vis_i = cv2.cvtColor(grid_vis_i, cv2.COLOR_BGR2RGB)
    html.save_image([grid_vis_i], [''], header='epoch_%3.3d' % n_epochs, width=grid_vis.shape[1], cvt=False)
    html.save()
    n_epochs += 1
    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - args.lr / niter_decay))

    dcgan_theano.save_model(disc_params, '%s/disc_params_%3.3d' % (model_dir, n_epochs))
    dcgan_theano.save_model(gen_params, '%s/gen_params_%3.3d' % (model_dir, n_epochs))
