from lasagne.layers import InputLayer, Conv2DLayer
from lasagne.layers import MaxPool2DLayer, LocalResponseNormalization2DLayer
from lasagne.layers import SliceLayer, concat, DenseLayer
from lasagne.layers import Upscale2DLayer
import theano as T
import lasagne
import numpy as np
from lasagne.utils import floatX
import os
from lib.theano_utils import sharedX
from lib import utils
pkg_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(pkg_dir, '../models/')

def build_model(x=None, layer='fc8', shape=(None, 3, 227, 227), up_scale=4):
    net = {'data': InputLayer(shape=shape, input_var=x)}
    net['data_s'] = Upscale2DLayer(net['data'], up_scale)
    net['conv1'] = Conv2DLayer(
            net['data_s'],
            num_filters=96,
            filter_size=(11, 11),
            stride=4,
            nonlinearity=lasagne.nonlinearities.rectify)

    if layer is 'conv1':
        return net

    # pool1
    net['pool1'] = MaxPool2DLayer(net['conv1'], pool_size=(3, 3), stride=2)

    # norm1
    net['norm1'] = LocalResponseNormalization2DLayer(net['pool1'],
                                                     n=5,
                                                     alpha=0.0001 / 5.0,
                                                     beta=0.75,
                                                     k=1)

    # conv2
    # before conv2 split the data
    net['conv2_data1'] = SliceLayer(net['norm1'], indices=slice(0, 48), axis=1)
    net['conv2_data2'] = SliceLayer(net['norm1'], indices=slice(48, 96), axis=1)

    # now do the convolutions
    net['conv2_part1'] = Conv2DLayer(net['conv2_data1'],
                                     num_filters=128,
                                     filter_size=(5, 5),
                                     pad=2)
    net['conv2_part2'] = Conv2DLayer(net['conv2_data2'],
                                     num_filters=128,
                                     filter_size=(5, 5),
                                     pad=2)

    # now combine
    net['conv2'] = concat((net['conv2_part1'], net['conv2_part2']), axis=1)
    if layer is 'conv2':
        return net
    # pool2
    net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=(3, 3), stride=2)

    # norm2
    net['norm2'] = LocalResponseNormalization2DLayer(net['pool2'],
                                                     n=5,
                                                     alpha=0.0001 / 5.0,
                                                     beta=0.75,
                                                     k=1)
    # conv3
    # no group
    net['conv3'] = Conv2DLayer(net['norm2'],
                               num_filters=384,
                               filter_size=(3, 3),
                               pad=1)
    if layer is 'conv3':
        return net

    # conv4
    net['conv4_data1'] = SliceLayer(net['conv3'], indices=slice(0, 192), axis=1)
    net['conv4_data2'] = SliceLayer(net['conv3'], indices=slice(192, 384), axis=1)
    net['conv4_part1'] = Conv2DLayer(net['conv4_data1'],
                                     num_filters=192,
                                     filter_size=(3, 3),
                                     pad=1)
    net['conv4_part2'] = Conv2DLayer(net['conv4_data2'],
                                     num_filters=192,
                                     filter_size=(3, 3),
                                     pad=1)
    net['conv4'] = concat((net['conv4_part1'], net['conv4_part2']), axis=1)
    if layer is 'conv4':
        return net

    # conv5
    # group 2
    net['conv5_data1'] = SliceLayer(net['conv4'], indices=slice(0, 192), axis=1)
    net['conv5_data2'] = SliceLayer(net['conv4'], indices=slice(192, 384), axis=1)
    net['conv5_part1'] = Conv2DLayer(net['conv5_data1'],
                                     num_filters=128,
                                     filter_size=(3, 3),
                                     pad=1)
    net['conv5_part2'] = Conv2DLayer(net['conv5_data2'],
                                     num_filters=128,
                                     filter_size=(3, 3),
                                     pad=1)
    net['conv5'] = concat((net['conv5_part1'], net['conv5_part2']), axis=1)
    if layer is 'conv5':
        return net

    # pool 5
    net['pool5'] = MaxPool2DLayer(net['conv5'], pool_size=(3, 3), stride=2)

    # fc6
    net['fc6'] = DenseLayer(
            net['pool5'], num_units=4096,
            nonlinearity=lasagne.nonlinearities.rectify)
    if layer is 'fc6':
        return net

    # fc7
    net['fc7'] = DenseLayer(
            net['fc6'],
            num_units=4096,
            nonlinearity=lasagne.nonlinearities.rectify)
    if layer is 'fc7':
        return net

    # fc8
    net['fc8'] = DenseLayer(
            net['fc7'],
            num_units=1000,
            nonlinearity=lasagne.nonlinearities.softmax)
    if layer is 'fc8':
        # st()
        return net

def load_model(net, layer='fc8'):
    model_values = utils.PickleLoad(os.path.join(model_dir, 'caffe_reference_%s.pkl' % layer))
    lasagne.layers.set_all_param_values(net[layer], model_values)



def transform_im(x, npx=64, nc=3):
    if nc == 3:
        x1 = (x + sharedX(1.0)) * sharedX(127.5)
    else:
        x1 = T.tile(x, [1,1,1,3]) * sharedX(255.0)  #[hack] to-be-tested

    mean_channel = np.load(os.path.join(pkg_dir, 'ilsvrc_2012_mean.npy')).mean(1).mean(1)
    mean_im = mean_channel[np.newaxis,:,np.newaxis,np.newaxis]
    mean_im = floatX(np.tile(mean_im, [1,1, npx, npx]))
    x2 = x1[:, [2,1,0], :,:]
    y = x2 - mean_im
    return y
