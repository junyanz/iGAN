from theano.sandbox.cuda.dnn import dnn_conv
from lib import activations
from lib import inits
from lib.ops import batchnorm, deconv
from lib.theano_utils import floatX, sharedX
from lib import utils

import theano
import theano.tensor as T
import cv2
from lib.theano_utils import floatX
from lib.rng import np_rng
import numpy as np



relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()

gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)

def save_image(im, filepath):
    tmp = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filepath, tmp)

def save_model(params, model_path):
    utils.PickleSave(model_path, [param.get_value() for param in params])



def set_model(params, params_values):
  for p, v in zip(params, params_values):
        p.set_value(v)


def load_model(params, model_path):
    param_values = utils.PickleLoad(model_path)
    set_model(params, param_values)
    return

def load_batchnorm(model_path):
    bn = utils.PickleLoad(model_path)
    bn_params = [sharedX(b) for b in bn]
    return bn_params

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
    x = deconv(hs[-1], _params[-1], subsample=(2, 2), border_mode=(2, 2))

    if nc == 3:
        x_f = tanh(x)
    if nc == 1:
        x_f = sigmoid(x)
    return x_f


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



def transform(x, nc=3):
    if nc == 3:
        return floatX(x).transpose(0, 3, 1, 2) / 127.5 - 1.
    else:
        return floatX(x).transpose(0, 3, 1, 2) / 255.0



def inverse_transform(x, npx=64, nc=3):
    if nc == 3:
        return (x.reshape(-1, 3, npx, npx).transpose(0, 2, 3, 1) + 1.) / 2.
    else:
        return 1.0 - x.reshape(-1, 1, npx, npx).transpose(0, 2, 3, 1)



def init_gen_params(nz=100, n_f=128, n_layers=3, init_sz=4, fs=5, nc=3):
    print('n_layers=', n_layers)
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



def init_disc_params(n_f=128, n_layers=3, init_sz=4, fs=5, nc=3):
    all_params = []
    dw0 = difn((n_f, nc, fs, fs), 'dw0')
    all_params.extend([dw0])

    for n in range(n_layers):
        outputf = n_f * 2 ** (n + 1)
        inputf = n_f * 2 ** n
        dw = difn((outputf, inputf, fs, fs), 'dw%d' % (n + 1))
        dg = gain_ifn((outputf), 'dg%d' % (n + 1))
        db = bias_ifn((outputf), 'db%d' % (n + 1))
        all_params.extend([dw, dg, db])  # 3* n_layers

    dwy = difn((n_f * 2 ** n_layers * init_sz * init_sz, 1), 'dwy')
    all_params.extend([dwy])
    return all_params



def gen_batchnorm(_z, _params, n_layers=3, n_f=128, init_sz=4, nc=3):
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

    if nc == 3:
        x = tanh(deconv(hs[-1], _params[-1], subsample=(2, 2), border_mode=(2, 2)))
    if nc == 1:
        x = sigmoid(deconv(hs[-1], _params[-1], subsample=(2, 2), border_mode=(2, 2)))

    return x, output


def discrim_batchnorm(_x, _params, n_layers=3):
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


def predict_batchnorm(_x, _params, n_layers=3):
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


    return y, output


def gen_test(_z, _params, _bn, n_layers=3, n_f=128, init_sz=4):
    [gw0, gg0, gb0] = _params[0:3]
    hs = []
    u = _bn[0]
    s = _bn[n_layers + 1]
    h0 = relu(batchnorm(T.dot(T.clip(_z, -1.0, 1.0), gw0), u=u, s=s, g=gg0, b=gb0))
    h1 = h0.reshape((h0.shape[0], n_f * 2 ** n_layers, init_sz, init_sz))
    hs.extend([h0, h1])
    for n in range(n_layers):
        [w, g, b] = _params[3 * (n + 1):3 * (n + 2)]
        hin = hs[-1]
        u = _bn[n + 1]
        s = _bn[n + n_layers + 2]
        hout = relu(batchnorm(deconv(hin, w, subsample=(2, 2), border_mode=(2, 2)), u=u, s=s, g=g, b=b))
        hs.append(hout)
    x = tanh(deconv(hs[-1], _params[-1], subsample=(2, 2), border_mode=(2, 2)))
    return x

