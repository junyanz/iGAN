from theano.sandbox.cuda.dnn import dnn_conv
from lib import activations
from lib import inits
from lib.ops import batchnorm, deconv
from itertools import izip
from lib.theano_utils import floatX, sharedX
from lib import utils

import theano
import theano.tensor as T
from time import time
from lib.theano_utils import floatX
from lib.rng import np_rng
import numpy as np
import dcgan_theano_config

class Model(object):
    def __init__(self, model_name, model_file):
        self.model_name = model_name
        self.model_file = model_file
        self.npx, self.n_layers, self.n_f, self.nc = getattr(dcgan_theano_config, model_name)()
        self.disc_params, self.gen_params, self.disc_pl, self.gen_pl = get_params(model_file, self.n_layers, self.n_f)
        # compile gen
        self._gen = self.def_gen(self.gen_params, self.gen_pl, self.n_layers, self.n_f)

    def model_G(self, z):  # z => x
        return gen_test_tanh(z, self.gen_params, self.gen_pl, self.n_layers, self.n_f)

    def model_D(self, x):  # x => 0/1
        return disc_test(x, self.disc_params, self.disc_pl, n_layers=self.n_layers)

    def def_gen(self, gen_params, gen_pl, n_layers, n_f):
        z = T.matrix()
        gx = gen_test(z, gen_params, gen_pl, n_layers=n_layers, n_f=n_f)
        print 'COMPILING...'
        t = time()
        _gen = theano.function([z], gx)
        print '%.2f seconds to compile _gen function' % (time() - t)
        return _gen

    def gen_samples(self, z0=None, n=32, batch_size=32, nz=100,  use_transform=True):
        assert n % batch_size == 0

        samples = []

        if z0 is None:
            z0 = np_rng.uniform(-1., 1., size=(n, nz))
        else:
            n = len(z0)
            batch_size = max(n, 64)
        n_batches = int(np.ceil(n/float(batch_size)))

        for i in range(n_batches):
            zmb = floatX(z0[batch_size * i:min(len(z0), batch_size * (i + 1)), :])
            xmb = self._gen(zmb)
            samples.append(xmb)

        samples = np.concatenate(samples, axis=0)

        if use_transform:
            samples = self.inverse_transform(samples, npx=self.npx)
            samples = (samples * 255).astype(np.uint8)
        return samples


    def transform(self, x):
        return floatX(x).transpose(0, 3, 1, 2) / 127.5 - 1.


    def transform_gray(self, x):
        return floatX(x).transpose(0, 3, 1, 2) / 255.0

    def transform_mask(self, x):
        return floatX(x).transpose(0, 3, 1, 2) / 255.0

    def inverse_transform(self, x, npx=64):
        x = (x.reshape(-1, 3, npx, npx).transpose(0, 2, 3, 1) + 1.) / 2.
        return x

    def inverse_transform_gray(self, x, npx=64):
        x = 1.0 - x.reshape(-1, 1, npx, npx).transpose(0, 2, 3, 1)
        return x



def get_params(model_file, n_layers, n_f, nz=100, nc=3):
    print 'LOADING...'
    t = time()

    disc_params = init_disc_params(nz=nz, n_f=n_f, n_layers=n_layers, nc=nc)
    gen_params = init_gen_params(nz=nz, n_f=n_f, n_layers=n_layers, nc=nc)
    # load the model
    model = utils.PickleLoad(model_file)
    print 'load model from %s' % model_file
    set_model(disc_params, model['disc_params'])
    set_model(gen_params, model['gen_params'])
    [disc_pl, gen_pl] = model['postlearn_params']
    disc_pl = [sharedX(d) for d in disc_pl]
    gen_pl = [sharedX(d) for d in gen_pl]
    print '%.2f seconds to load theano models' % (time() - t)
    return disc_params, gen_params, disc_pl, gen_pl


def set_model(params, params_values):
  for p, v in izip(params, params_values):
        p.set_value(v)

def reset_adam(_updates):
    for n, update in enumerate(_updates):
        value = update[0].get_value()
        if n == 2:
            continue
        if n == 3:
            update[0].set_value(floatX(1.0))
            continue
        update[0].set_value(floatX(np.zeros_like(value)))


relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()

gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)


# print(n_layers)
def init_gen_params(nz=100, n_f=128, n_layers=3, init_sz=4, fs=5, nc=3):
    # nz to flatten
    print('n_layers=', n_layers)
    # st()
    gen_params = []
    outputf = n_f * 2 ** n_layers * init_sz * init_sz
    gw0 = gifn((nz, outputf), 'gw0')
    gg0 = gain_ifn((outputf), 'gg0')
    gb0 = bias_ifn((outputf), 'gb0')
    gen_params.extend([gw0, gg0, gb0])
    for n in range(0, n_layers):
        inputf = n_f * 2 ** (n_layers - n)
        outputf = n_f * 2 ** (n_layers - n - 1)
        gw = gifn((inputf, outputf, fs, fs), 'gw%d' % (n + 1))
        gg = gain_ifn((outputf), 'gg%d' % (n + 1))
        gb = bias_ifn((outputf), 'gb%d' % (n + 1))
        gen_params.extend([gw, gg, gb])
    gwx = gifn((n_f, nc, fs, fs), 'gwx')
    gen_params.append(gwx)
    return gen_params
#
#
def init_predict_params(nz=100, n_f=128, n_layers=3, init_sz=4, fs=5, nc=3):
    disc_params = []
    dw0 = difn((n_f, nc, fs, fs), 'dw0')
    disc_params.append(dw0)
    for n in range(n_layers):
        outputf = n_f * 2 ** (n + 1)
        inputf = n_f * 2 ** n
        dw = difn((outputf, inputf, fs, fs), 'dw%d' % (n + 1))
        dg = gain_ifn((outputf), 'dg%d' % (n + 1))
        db = bias_ifn((outputf), 'db%d' % (n + 1))
        disc_params.extend([dw, dg, db])
    dwy = difn((n_f * 2 ** n_layers * init_sz * init_sz, nz), 'dwy')
    disc_params.append(dwy)
    return disc_params


def gen(_z, _params, n_layers=3, n_f=128, init_sz=4, nc=3):
    [gw0, gg0, gb0] = _params[0:3]
    hs = []
    h0 = relu(batchnorm(T.dot(_z, gw0), g=gg0, b=gb0))
    h1 = h0.reshape((h0.shape[0], n_f * 2 ** n_layers, init_sz, init_sz))
    hs.extend([h0, h1])
    for n in range(n_layers):
        [w, g, b] = _params[3 * (n + 1):3 * (n + 2)]
        hin = hs[-1]
        hout = relu(batchnorm(deconv(hin, w, subsample=(2, 2), border_mode=(2, 2)), g=g, b=b))
        hs.append(hout)
    if nc == 3:
        x = tanh(deconv(hs[-1], _params[-1], subsample=(2, 2), border_mode=(2, 2)))
    if nc == 1:
        x = sigmoid(deconv(hs[-1], _params[-1], subsample=(2, 2), border_mode=(2, 2)))
    return x


def discrim(_x, _params, n_layers=3):
    w = _params[0]
    h0 = lrelu(dnn_conv(_x, w, subsample=(2, 2), border_mode=(2, 2)))
    hs = [h0]
    for n in range(n_layers):
        hin = hs[-1]
        w, g, b = _params[1 + 3 * n: 1 + 3 * (n + 1)]
        hout = lrelu(batchnorm(dnn_conv(hin, w, subsample=(2, 2), border_mode=(2, 2)), g=g, b=b))
        hs.append(hout)
    h = T.flatten(hs[-1], 2)
    y = sigmoid(T.dot(h, _params[-1]))
    return y


def init_disc_params(nz=100, n_f=128, n_layers=3, init_sz=4, fs=5, nc=3):
    all_params = []
    dw0 = difn((n_f, nc, fs, fs), 'dw0')
    # db0 = bias_ifn((n_f), 'db0')
    all_params.extend([dw0])#, db0])  # 2

    for n in range(n_layers):
        outputf = n_f * 2 ** (n + 1)
        inputf = n_f * 2 ** n
        dw = difn((outputf, inputf, fs, fs), 'dw%d' % (n + 1))
        dg = gain_ifn((outputf), 'dg%d' % (n + 1))
        db = bias_ifn((outputf), 'db%d' % (n + 1))
        all_params.extend([dw, dg, db])  # 3* n_layers

    dwy = difn((n_f * 2 ** n_layers * init_sz * init_sz, 1), 'dwy')
    # dby = bias_ifn((1), 'dby')
    all_params.extend([dwy])#,dby])#2
    return all_params


def discrim(_x, _params, n_layers=3):
    w0 = _params[0]
    # b0 = _params[1]
    h0 = lrelu(dnn_conv(_x, w0, subsample=(2, 2), border_mode=(2, 2)))#+ b0.dimshuffle('x', 0, 'x', 'x'))
    hs = [h0]
    for n in range(n_layers):
        hin = hs[-1]
        w, g, b = _params[1 + 3 * n: 1 + 3 * (n + 1)]
        hout = lrelu(batchnorm(dnn_conv(hin, w, subsample=(2, 2), border_mode=(2, 2)), g=g, b=b))
        hs.append(hout)
    h = T.flatten(hs[-1], 2)
    yw = _params[-1]
    # yb = _params[-1]
    y = sigmoid(T.dot(h, yw))
    return y


def predict(_x, _params, n_layers=3):
    w = _params[0]
    h0 = lrelu(dnn_conv(_x, w, subsample=(2, 2), border_mode=(2, 2)))
    hs = [h0]
    for n in range(n_layers):
        hin = hs[-1]
        w, g, b = _params[1 + 3 * n:1 + 3 * (n + 1)]
        hout = lrelu(batchnorm(dnn_conv(hin, w, subsample=(2, 2), border_mode=(2, 2)), g=g, b=b))
        hs.append(hout)
    h = T.flatten(hs[-1], 2)
    y = tanh(T.dot(h, _params[-1]))
    return y


def gen_postlearn(_z, _params, n_layers=3, n_f=128, init_sz=4):
    output = []
    [gw0, gg0, gb0] = _params[0:3]
    hs = []
    h0_o = T.dot(_z, gw0)
    output = [h0_o]
    h0 = relu(batchnorm(h0_o, g=gg0, b=gb0))
    h1 = h0.reshape((h0.shape[0], n_f * 2 ** n_layers, init_sz, init_sz))
    hs.extend([h0, h1])
    for n in range(n_layers):
        [w, g, b] = _params[3 * (n + 1):3 * (n + 2)]
        hin = hs[-1]
        h_o = deconv(hin, w, subsample=(2, 2), border_mode=(2, 2))
        hout = relu(batchnorm(h_o, g=g, b=b))
        hs.append(hout)
        output.append(h_o)
    x = tanh(deconv(hs[-1], _params[-1], subsample=(2, 2), border_mode=(2, 2)))
    return x, output


def discrim_postlearn(_x, _params, n_layers=3):
    w = _params[0]
    h0 = lrelu(dnn_conv(_x, w, subsample=(2, 2), border_mode=(2, 2)))
    hs = [h0]
    output = []
    for n in range(n_layers):
        hin = hs[-1]
        w, g, b = _params[1 + 3 * n:1 + 3 * (n + 1)]
        h_o = dnn_conv(hin, w, subsample=(2, 2), border_mode=(2, 2))
        hout = lrelu(batchnorm(h_o, g=g, b=b))
        hs.append(hout)
        output.append(h_o)
    h = T.flatten(hs[-1], 2)
    y = sigmoid(T.dot(h, _params[-1]))
    return y, output


def predict_postlearn(_x, _params, n_layers=3):
    w = _params[0]
    h0 = lrelu(dnn_conv(_x, w, subsample=(2, 2), border_mode=(2, 2)))
    hs = [h0]
    output = []
    for n in range(n_layers):
        hin = hs[-1]
        w, g, b = _params[1 + 3 * n:1 + 3 * (n + 1)]
        h_o = dnn_conv(hin, w, subsample=(2, 2), border_mode=(2, 2))
        hout = lrelu(batchnorm(h_o, g=g, b=b))
        hs.append(hout)
        output.append(h_o)
    h = T.flatten(hs[-1], 2)
    y = tanh(T.dot(h, _params[-1]))
    return y, output


def gen_test(_z, _params, _pls, n_layers=3, n_f=128, init_sz=4):
    [gw0, gg0, gb0] = _params[0:3]
    hs = []
    u = _pls[0]
    s = _pls[n_layers + 1]
    h0 = relu(batchnorm(T.dot(T.clip(_z, -1.0, 1.0), gw0), u=u, s=s, g=gg0, b=gb0))
    h1 = h0.reshape((h0.shape[0], n_f * 2 ** n_layers, init_sz, init_sz))
    hs.extend([h0, h1])
    for n in range(n_layers):
        [w, g, b] = _params[3 * (n + 1):3 * (n + 2)]
        hin = hs[-1]
        u = _pls[n + 1]
        s = _pls[n + n_layers + 2]
        hout = relu(batchnorm(deconv(hin, w, subsample=(2, 2), border_mode=(2, 2)), u=u, s=s, g=g, b=b))
        hs.append(hout)
    x = tanh(deconv(hs[-1], _params[-1], subsample=(2, 2), border_mode=(2, 2)))
    return x


def gen_test_tanh(_z, _params, _pls, n_layers=3, n_f=128, init_sz=4):
    tan_z = tanh(_z)
    [gw0, gg0, gb0] = _params[0:3]
    hs = []
    u = _pls[0]
    s = _pls[n_layers + 1]
    h0 = relu(batchnorm(T.dot(T.clip(tan_z, -1.0, 1.0), gw0), u=u, s=s, g=gg0, b=gb0))
    h1 = h0.reshape((h0.shape[0], n_f * 2 ** n_layers, init_sz, init_sz))
    hs.extend([h0, h1])
    for n in range(n_layers):
        [w, g, b] = _params[3 * (n + 1):3 * (n + 2)]
        hin = hs[-1]
        u = _pls[n + 1]
        s = _pls[n + n_layers + 2]
        hout = relu(batchnorm(deconv(hin, w, subsample=(2, 2), border_mode=(2, 2)), u=u, s=s, g=g, b=b))
        hs.append(hout)
    x = tanh(deconv(hs[-1], _params[-1], subsample=(2, 2), border_mode=(2, 2)))
    return x


def disc_test(_x, _params, _pls, n_layers=3):
    w = _params[0]
    h0 = lrelu(dnn_conv(_x, w, subsample=(2, 2), border_mode=(2, 2)))
    hs = [h0]
    for n in range(n_layers):
        hin = hs[-1]
        w, g, b = _params[1 + 3 * n:1 + 3 * (n + 1)]
        u = _pls[n]
        s = _pls[n + n_layers]
        hout = lrelu(batchnorm(dnn_conv(hin, w, subsample=(2, 2), border_mode=(2, 2)), u=u, s=s, g=g, b=b))
        hs.append(hout)
    h = T.flatten(hs[-1], 2)
    y = sigmoid(T.dot(h, _params[-1]))
    return y


def predict_test(_x, _params, _pls, n_layers=3):
    w = _params[0]
    h0 = lrelu(dnn_conv(_x, w, subsample=(2, 2), border_mode=(2, 2)))
    hs = [h0]
    for n in range(n_layers):
        hin = hs[-1]
        w, g, b = _params[1 + 3 * n:1 + 3 * (n + 1)]
        u = _pls[n]
        s = _pls[n + n_layers]
        hout = lrelu(batchnorm(dnn_conv(hin, w, subsample=(2, 2), border_mode=(2, 2)), u=u, s=s, g=g, b=b))
        hs.append(hout)
    h = T.flatten(hs[-1], 2)
    y = tanh(T.dot(h, _params[-1]))
    return y
