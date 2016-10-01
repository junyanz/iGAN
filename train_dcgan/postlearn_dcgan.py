import sys
sys.path.append('..')
import numpy as np
from tqdm import tqdm
import cPickle
import theano
import theano.tensor as T
from lib import config
from lib.rng import np_rng
from lib.theano_utils import floatX
from lib import modeldef
# import modeldef
batch_size = 128      # # of examples in batch
nz = 100

n_batch = 1000
# n_f = 128
# n_layers = 3
desc = 'handbag_sketch_64'
ext = ''
expr_name = desc + ext
# st()
npx, n_layers, n_f, c_iter, nc = getattr(config, desc)()
model_dir = '../../models/%s' % expr_name

disc_params = modeldef.init_disc_params(nz=nz, n_f=n_f, n_layers=n_layers, nc=1)
gen_params = modeldef.init_gen_params(nz=nz, n_f=n_f, n_layers=n_layers, nc=1)


print('load model from %s, c_iter=%d, expr_name=%s' % (model_dir,c_iter, expr_name))
modeldef.load_model(disc_params, '%s/disc_params_%3.3d' % (model_dir, c_iter))
modeldef.load_model(gen_params, '%s/gen_params_%3.3d' % (model_dir, c_iter))

Z = T.matrix()
gX, gbn = modeldef.gen_postlearn(Z, gen_params, n_layers=n_layers, n_f=n_f)
p_gen, dbn = modeldef.discrim_postlearn(gX, disc_params, n_layers=n_layers)
ngbn = len(gbn)
ndbn = len(dbn)
bn_data = gbn + dbn
_postlearn = theano.function([Z], bn_data)

nb_sum = []
nb_mean = []
nb_mean_ext = []


# first pass
for n in tqdm(range(n_batch)):
    zmb = floatX(np_rng.uniform(-1., 1., size=(batch_size, nz)))
    bn_data = _postlearn(zmb)

    if n == 0:
        for d in bn_data:
            nb_sum.append(d)
    else:
        for id, d in enumerate(bn_data):
            nb_sum[id] = nb_sum[id] + d

# compute empirical mean
for id, d_sum in enumerate(nb_sum):
    if d_sum.ndim == 4:
        m = np.mean(d_sum, axis=(0, 2, 3)) / n_batch  # .dimshuffle('x', 0, 'x', 'x')
        nb_mean.append(m)
        nb_mean_ext.append(np.reshape(m, [1, len(m), 1, 1]))
    if d_sum.ndim == 2:
        m = np.mean(d_sum, axis=0) / n_batch
        nb_mean.append(m)
        nb_mean_ext.append(m)


# second pass
nb_var_sum = []

for n in tqdm(range(n_batch)):
    zmb = floatX(np_rng.uniform(-1., 1., size=(batch_size, nz)))
    bn_data = _postlearn(zmb)
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
        nb_var.append(np.mean(var_sum, axis=(0, 2, 3)) / (n_batch - 1))

    if var_sum.ndim == 2:
        nb_var.append(np.mean(var_sum, axis=0) / (n_batch - 1))
modelpath = '%s/%spostlearn_%3.3d' % (model_dir, ext, c_iter)
with open(modelpath, "wb") as f:
    cPickle.dump(ngbn, f, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(nb_mean, f, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(nb_var, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
