from __future__ import print_function
import sys

sys.path.append('..')
import os
import json
import numpy as np
from tqdm import tqdm
import theano
import theano.tensor as T
from lib import utils
from lib import updates
from lib import costs
from lib.theano_utils import floatX, sharedX
from load import load_imgs
from lib import image_save
import train_dcgan_utils
import train_dcgan_config
from lib import AlexNet
import lasagne
import argparse
from time import time
import cv2
from pdb import set_trace as st

# set arguments and parameters
parser = argparse.ArgumentParser('Train a predictive network (x->z) to invert the generative network')
parser.add_argument('--model_name', dest='model_name', help='model name', default='shoes_64', type=str)
parser.add_argument('--ext', dest='ext', help='experiment name=model_name+ext', default='', type=str)
parser.add_argument('--data_file', dest='data_file', help='the file that stores the hdf5 data', type=str, default=None)
parser.add_argument('--cache_dir', dest='cache_dir', help='cache directory that stores models, samples and webpages', type=str, default=None)
parser.add_argument('--batch_size', dest='batch_size', help='the number of examples in each batch', type=int, default=128)

parser.add_argument('--save_freq', dest='save_freq', help='save a model every save_freq epochs', type=int, default=1)
parser.add_argument('--lr', dest='lr', help='learning rate', type=float, default=0.0002)
parser.add_argument('--weight_decay', dest='weight_decay', help='l2 weight decay', type=float, default=1e-5)
parser.add_argument('--b1', dest='b1', help='momentum term of adam', type=float, default=0.5)
parser.add_argument('--layer', dest='layer', help='the AlexNet layer used for feature loss', type=str, default='conv4')
parser.add_argument('--alpha', dest='alpha', help='the weight for feature loss. Loss=pixel_loss + alpha * feature_loss', type=float, default=0.002)
args = parser.parse_args()


if not args.data_file:
    args.data_file = '../datasets/%s.hdf5' % args.model_name

npx, n_layers, n_f, nc, nz, niter, niter_decay = getattr(train_dcgan_config, args.model_name)()
expr_name = args.model_name + args.ext

if not args.cache_dir:
    args.cache_dir = './cache/%s/' % expr_name

for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

# create directories
rec_dir = os.path.join(args.cache_dir, 'rec')
model_dir = os.path.join(args.cache_dir, 'models')
log_dir = os.path.join(args.cache_dir, 'log')
web_dir = os.path.join(args.cache_dir, 'web_rec')
html = image_save.ImageSave(web_dir, expr_name, append=True)
utils.mkdirs([rec_dir, model_dir, log_dir, web_dir])

# load data
tr_data, te_data, tr_stream, te_stream, ntrain, ntest \
    = load_imgs(ntrain=None, ntest=None, batch_size=args.batch_size, data_file=args.data_file)
te_handle = te_data.open()
ntest = int(np.floor(ntest/float(args.batch_size)) * args.batch_size)
# st()
test_x, = te_data.get_data(te_handle, slice(0, ntest))

test_x = train_dcgan_utils.transform(test_x, nc=nc)
predict_params = train_dcgan_utils.init_predict_params(nz=nz, n_f=n_f, n_layers=n_layers, nc=nc)
# load modelG
gen_params = train_dcgan_utils.init_gen_params(nz=nz, n_f=n_f, n_layers=n_layers, nc=nc)
train_dcgan_utils.load_model(gen_params, os.path.join(model_dir, 'gen_params'))
gen_batchnorm = train_dcgan_utils.load_batchnorm(os.path.join(model_dir, 'gen_batchnorm'))

# define the model
t= time()
x = T.tensor4()
z = train_dcgan_utils.predict(x, predict_params, n_layers=n_layers)
gx = train_dcgan_utils.gen_test(z, gen_params, gen_batchnorm, n_layers=n_layers, n_f=n_f)

# define pixel loss
pixel_loss = costs.L2Loss(gx, x)

# define feature loss
x_t = AlexNet.transform_im(x, npx=npx, nc=nc)
x_net = AlexNet.build_model(x_t, layer=args.layer, shape=(None, 3, npx, npx))
AlexNet.load_model(x_net, layer=args.layer)
x_f = lasagne.layers.get_output(x_net[args.layer], deterministic=True)
gx_t = AlexNet.transform_im(gx, npx=npx, nc=nc)
gx_net = AlexNet.build_model(gx_t, layer=args.layer, shape=(None, 3, npx, npx))
AlexNet.load_model(gx_net, layer=args.layer)
gx_f = lasagne.layers.get_output(gx_net[args.layer], deterministic=True)
ftr_loss = costs.L2Loss(gx_f, x_f)

# add two losses together
cost = pixel_loss + ftr_loss * sharedX(args.alpha)
output = [cost, z]
lrt = sharedX(args.lr)
b1t = sharedX(args.b1)
p_updater = updates.Adam(lr=lrt, b1=b1t, regularizer=updates.Regularizer(l2=args.weight_decay))
p_updates = p_updater(predict_params, cost)

print('COMPILING')
t = time()
_train_p = theano.function([x], cost, updates=p_updates)
_train_p_cost = theano.function([x], [cost, gx])
_predict_z = theano.function([x], z)
_gen = theano.function([z], gx)
print('%.2f seconds to compile theano functions' % (time() - t))


def rec_test(test_data, n_epochs=0, batch_size=128, output_dir=None):

    print('computing reconstruction loss on test images')
    rec_imgs = []
    imgs = []
    costs = []
    ntest = len(test_data)

    for n in tqdm(range(ntest / batch_size)):
        imb = test_data[n*batch_size:(n+1)*batch_size, ...]
        # imb = train_dcgan_utils.transform(xmb, nc=3)
        [cost, gx] = _train_p_cost(imb)
        costs.append(cost)
        ntest = ntest + 1
        if n == 0:
            utils.print_numpy(imb)
            utils.print_numpy(gx)
            imgs.append(train_dcgan_utils.inverse_transform(imb, npx=npx, nc=nc))
            rec_imgs.append(train_dcgan_utils.inverse_transform(gx, npx=npx, nc=nc))

    if output_dir is not None:
        # st()
        save_samples = np.hstack(np.concatenate(imgs, axis=0))
        save_recs = np.hstack(np.concatenate(rec_imgs, axis=0))
        save_comp = np.vstack([save_samples, save_recs])
        mean_cost = np.mean(costs)

        txt = 'epoch = %3.3d, cost = %3.3f' % (n_epochs, mean_cost)

        width = save_comp.shape[1]
        save_f = (save_comp*255).astype(np.uint8)
        html.save_image([save_f], [''], header=txt, width=width, cvt=True)
        html.save()
        save_cvt = cv2.cvtColor(save_f, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(rec_dir, 'rec_epoch_%5.5d.png'%n_epochs), save_cvt)

    return mean_cost


f_log = open('%s/training_predict_log.ndjson' % log_dir, 'wb')
log_fields = ['n_epochs', 'n_updates', 'n_examples', 'n_seconds', 'test_cost']

n_epochs = 0
n_updates = 0
n_examples = 0
t = time()

for epoch in range(niter + niter_decay):
    n = 0
    total_n = int(ntrain / args.batch_size)
    for xmb, in tr_stream.get_epoch_iterator():
        imb = train_dcgan_utils.transform(xmb, nc=nc)
        n += 1
        train_cost = _train_p(imb)
        print('epoch = %3.3d, n = %3.3d/%3.3d, train_cost = %4.4f' % (epoch, n, total_n, train_cost)),
        n_updates += 1
        n_examples += len(xmb)

    n_epochs += 1

    if n_epochs % args.save_freq == 0:
        train_dcgan_utils.save_model(predict_params, '%s/predict_params%3.3d' % (model_dir, n_epochs))
    train_dcgan_utils.save_model(predict_params, os.path.join(model_dir, 'predict_params')) # save the latest one

    test_cost = rec_test(test_data=test_x, n_epochs=n_epochs, batch_size=args.batch_size, output_dir=rec_dir)
    log = [n_epochs, n_updates, n_examples, time() - t, float(test_cost)]

    print('%.0f %.4f' % (epoch, test_cost))
    f_log.write(json.dumps(dict(zip(log_fields, log))) + '\n')
    f_log.flush()

    if n_epochs > niter:
        lrt.set_value(floatX(lrt.get_value() - args.lr / niter_decay))
