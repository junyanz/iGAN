import sys
sys.path.append('..')
import numpy as np
from tqdm import tqdm
import cPickle
import theano
import theano.tensor as T
import modeldef
from lib.load import load_imgs
from lib import config
import modeldef

batch_size = 128      # # of examples in batch
nz = 100
desc = 'amazon_handbag'
ext=''
expr_name = desc+'_'+ext
npx, n_layers, n_f, c_iter = getattr(config, desc)()
# desc_z = desc + '_z'
model_dir = '../../models/%s' % expr_name
data_dir = '../../dataset/%s/images.hdf5' % desc
layer = 'conv4'
n_batch =1000
ntrain = batch_size*n_batch
tr_data, te_data, tr_stream, te_stream, ntrain, ntest = load_imgs(ntrain=ntrain, ntest=None, batch_size=batch_size,data_dir=data_dir)


predict_params = modeldef.init_predict_params(nz=nz, n_f=n_f, n_layers=n_layers)
print('load model from %s' % model_dir)
modeldef.load_model(predict_params, '%s/predict_params_%s' % (model_dir, layer))

x = T.tensor4()
z_p, bn_data = modeldef.predict_postlearn(x, predict_params, n_layers=n_layers)
_postlearn = theano.function([x], bn_data)

nb_sum = []
nb_mean = []
nb_mean_ext = []
n_batch = int(np.floor(ntrain / float(batch_size)))
print('n_batch = %d, batch_size = %d' % (n_batch, batch_size))
# st()
# first pass
# for n in tqdm(range(nBatch)):
n = 0
for imb, in tqdm(tr_stream.get_epoch_iterator(), total=ntrain / batch_size):
    imb = modeldef.transform(imb)
    bn_data = _postlearn(imb)

    if n == 0:
        for d in bn_data:
            nb_sum.append(d)
    else:
        for id, d in enumerate(bn_data):
            nb_sum[id] = nb_sum[id] + d
    n = n+1
    if n >= n_batch:
        break
# compute empirical mean
for id, d_sum in enumerate(nb_sum):
    if d_sum.ndim == 4:
        m = np.mean(d_sum, axis=(0, 2, 3)) / n_batch# .dimshuffle('x', 0, 'x', 'x')
        nb_mean.append(m)
        nb_mean_ext.append(np.reshape(m, [1, len(m), 1, 1]))
    if d_sum.ndim == 2:
        m = np.mean(d_sum, axis=0) / n_batch
        nb_mean.append(m)
        nb_mean_ext.append(m)


# second pass
tr_stream.reset()
nb_var_sum = []
n = 0

for imb, in tqdm(tr_stream.get_epoch_iterator(), total=ntrain / batch_size):
    imb = modeldef.transform(imb)
    bn_data = _postlearn(imb)
    if n == 0:
        for id, d in enumerate(bn_data):
            var = (np.copy(d) - nb_mean_ext[id]) ** 2
            nb_var_sum.append(var)
            # print('nb_var_sum =%d'%len(nb_var_sum))
    else:
        for id, d in enumerate(bn_data):
            var = (np.copy(d) - nb_mean_ext[id]) ** 2
            nb_var_sum[id] = nb_var_sum[id] + var
            # print('nb_var_sum =%d'%len(nb_var_sum))
    n += 1
    if n >= n_batch:
        break

# compute empirical variance
nb_var = []
for id, var_sum in enumerate(nb_var_sum):
    if var_sum.ndim == 4:
        nb_var.append(np.mean(var_sum, axis=(0, 2, 3)) / (n_batch- 1))

    if var_sum.ndim == 2:
        nb_var.append(np.mean(var_sum, axis=0) / (n_batch - 1))
# st()
modelpath = '%s/postlearn_predict_%s' % (model_dir,layer)
with open(modelpath, "wb") as f:
    # cPickle.dump(ngbn, f, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(nb_mean, f, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(nb_var, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
