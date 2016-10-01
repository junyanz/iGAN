import sys

sys.path.append('..')
import os
from time import time
import numpy as np
from tqdm import tqdm
import theano
import theano.tensor as T
from scipy.misc import imsave
from lib.theano_utils import floatX
from lib import modeldef
from lib.load import load_imgs
from lib.rng import np_rng
batch_size = 128  # # of examples in batch
npx = 64  # # of pixels width/height of images
nz = 100  # # of dim for Z
n_f = 128
n_layers = 3
n_vis = 196
desc = 'shoes_64'
data_dir = '../../dataset/%s.hdf5' % desc
model_dir = '../../models/%s' % desc
samples_dir = '../../samples/predict/%s' % desc
log_dir = '../../logs/%s' % desc
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

predict_params = modeldef.init_predict_params(nz=nz, n_f=n_f, n_layers=n_layers)
# load params
gen_params = modeldef.init_gen_params(nz=nz, n_f=n_f, n_layers=n_layers)
modeldef.load_model(gen_params, '%s/gen_params' % model_dir)
modeldef.load_model(predict_params, '%s/predict_params' % model_dir)
[disc_pl, gen_pl] = modeldef.load_pl('%s/postlearn' % model_dir)
predict_pl = modeldef.load_predict_pl('%s/predict_postlearn' % model_dir)
# define the model
x = T.tensor4()
z = T.matrix()

gx = modeldef.gen_test(z, gen_params, gen_pl, n_layers=n_layers, n_f=n_f)
p_z = modeldef.predict_test(x, predict_params, predict_pl, n_layers=n_layers)
print 'COMPILING'
t = time()

_gen = theano.function([z], gx)
_predict = theano.function([x], p_z)
print '%.2f seconds to compile _gen and _predict' % (time() - t)


def rec_test(te_stream=None, batch_size=128, samples_dir=None, real=True):
    print('rec_test')
    costs = []
    ntest = 0
    batch_id = 0
    for xmb, in tqdm(te_stream.get_epoch_iterator(), total=ntest / batch_size):
        # st()
        if real:
            xmb = modeldef.transform(xmb)
        else:
            zmb = floatX(np_rng.uniform(-1., 1., size=(batch_size, nz)))
            xmb = _gen(zmb)

        p_z = _predict(xmb)
        xmb_p = _gen(p_z)
        cost = np.mean((xmb-xmb_p) ** 2)
        costs.append(cost)
        ntest = ntest+1
        # st()
        if ntest <= 1:
            modeldef.print_x(xmb)
            modeldef.print_x(p_z)
            modeldef.print_x(xmb_p)

        if samples_dir is not None:
            imgs = np.hstack(modeldef.inverse_transform(xmb, npx=npx))
            rec_imgs = np.hstack(modeldef.inverse_transform(xmb_p, npx=npx))
            ext = real and 'real' or 'fake'
            save_comp = np.vstack([imgs, rec_imgs])
            imsave('%s/predict_%s_batch_%3.3d.png' % (samples_dir, ext, batch_id), save_comp)
        batch_id = batch_id+1
        # st()
    cost_mean = np.mean(costs)
    print('l2 costs = %5.5f' % cost_mean)

if __name__ == '__main__':
    tr_data, te_data, tr_stream, te_stream, ntrain, ntest \
    = load_imgs(ntrain=None, ntest=None, batch_size=batch_size,data_dir=data_dir)

    rec_test(te_stream, batch_size, samples_dir,real=True)
    rec_test(te_stream, batch_size, samples_dir,real=False)