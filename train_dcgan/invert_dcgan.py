import sys

sys.path.append('..')
import theano
import theano.tensor as T
from time import time
from tqdm import tqdm
from lib import updates, HOGNet
from lib.rng import np_rng
from lib.theano_utils import floatX, sharedX
import lib.modeldef
import numpy as np
from lib import utils
from lib.load import load_imgs_raw, load_imgs, load_imgs_seq
import lib.config
from lib import image_save
import os
import cPickle
from lib import alexnet
import lasagne
from train import gen_samples
from scipy import optimize

batch_size = 5
n_iters = 200
nz = 100


def def_predict(model_dir,desc, layer='conv4', nz=100):
    print 'LOADING...'
    t = time()
    npx, n_layers, n_f, c_iter = getattr(lib.config, desc)()
    predict_params = lib.modeldef.init_predict_params(nz=nz, n_f=n_f, n_layers=n_layers)
    print('load model from %s_%s' % (model_dir, layer))
    lib.modeldef.load_model(predict_params, '%s/predict_params_%s' % (model_dir, layer))
    pl_path = '%s/postlearn_predict_%s' % (model_dir, layer)
    with open(pl_path, 'rb') as f:
        bn_u = [sharedX(d) for d in cPickle.load(f)]
        bn_s = [sharedX(d) for d in cPickle.load(f)]
        predict_pl = bn_u + bn_s
    # st()
    print '%.2f seconds to load theano models' % (time() - t)
    print 'COMPILING...'
    t = time()

    x = T.tensor4()
    z = lib.modeldef.predict_test(x, predict_params, predict_pl, n_layers=n_layers)
    _predict = theano.function([x], [z])
    print '%.2f seconds to compile _predict function' % (time() - t)
    return _predict


def def_rec(layer='conv4', alpha=0.002):
    print 'COMPILING...'
    t = time()

    x = T.tensor4()
    y = T.tensor4()

    if layer is 'hog':
        x_f = HOGNet.get_hog(x, use_bin=True, BS=4)
        y_f = HOGNet.get_hog(y, use_bin=True, BS=4)
    else:
        x_t = alexnet.transform_im(x)
        x_net = alexnet.build_model(x_t, layer=layer, shape=(None, 3, 64, 64))
        alexnet.load_model(x_net, layer=layer)
        x_f = lasagne.layers.get_output(x_net[layer], deterministic=True)

        y_t = alexnet.transform_im(y)
        y_net = alexnet.build_model(y_t, layer=layer, shape=(None, 3, 64, 64))
        alexnet.load_model(y_net, layer=layer)
        y_f = lasagne.layers.get_output(y_net[layer], deterministic=True)

    f_rec = T.mean(T.sqr(x_f - y_f), axis=(1, 2, 3)) * sharedX(alpha)
    x_rec = T.mean(T.sqr(x - y), axis=(1, 2, 3))
    rec = f_rec + x_rec
    _rec = theano.function([x, y], [rec])
    print '%.2f seconds to compile def_rec_alexnet function' % (time() - t)
    return _rec

def def_psnr():
    x = T.tensor4()
    y = T.tensor4()
    x_rec = T.mean(T.sqr(x - y), axis=(1, 2, 3))
    const = sharedX(floatX(20 * np.log10(255.0)))
    psnr = const-sharedX(10)*T.log10(x_rec)
    _psnr = theano.function([x, y], [psnr,x_rec])
    return _psnr

def predict_z(_predict, ims, batch_size=32):
    ims_t = lib.modeldef.transform(ims)
    n = ims.shape[0]
    n_gen = 0
    zs = []
    n_batch = int(np.ceil(n / float(batch_size)))
    # st()
    for i in tqdm(range(n_batch)):
        imb = lib.modeldef.transform(ims[batch_size * i:min(n, batch_size * (i + 1)), :, :, :])
        # st()
        zmb = _predict(imb)
        zs.append(zmb)
        n_gen += len(imb)
    zs = np.squeeze(np.concatenate(zs, axis=0))
    if np.ndim(zs) == 1:
        zs = zs[np.newaxis, :]
    # st()
    return zs


def rec_loss(ims, recs, _rec, _psnr):
    ims_t = lib.modeldef.transform(ims)
    recs_t = lib.modeldef.transform(recs)
    loss_rec = _rec(ims_t, recs_t)
    ims_t2 = lib.modeldef.transform_no_scale(ims)
    recs_t2 = lib.modeldef.transform_no_scale(recs)
    loss_psnr,x_rec = _psnr(ims_t2, recs_t2)
    # st()
    return loss_rec, loss_psnr

# def psnr_loss(ims, recs, _rec)

def def_invert(model_dir, layer='conv4', alpha=0.002, lr=0.1, b1=0.5):
    npx, n_layers, n_f, c_iter = getattr(lib.config, desc)()
    [disc_params, gen_params, disc_pl, gen_pl] = lib.modeldef.get_params(model_dir, n_layers, n_f, c_iter)

    x_f = T.tensor4()
    x = T.tensor4()

    z = sharedX(floatX(np_rng.uniform(-1., 1., size=(batch_size, nz))))
    gx = lib.modeldef.gen_test_tanh(z, gen_params, gen_pl, n_layers=n_layers, n_f=n_f)

    # st()
    if layer is 'hog':
        gx_f = HOGNet.get_hog(gx, use_bin=True, BS=4)
    else:
        gx_t = alexnet.transform_im(gx)
        gx_net = alexnet.build_model(gx_t, layer=layer, shape=(None, 3, 64, 64))
        alexnet.load_model(gx_net, layer=layer)
        gx_f = lasagne.layers.get_output(gx_net[layer], deterministic=True)

    f_rec = T.mean(T.sqr(x_f - gx_f), axis=(1, 2, 3)) * sharedX(alpha)
    x_rec = T.mean(T.sqr(x - gx), axis=(1, 2, 3))
    cost = T.sum(f_rec) + T.sum(x_rec)

    d_updater = updates.Adam(lr=sharedX(lr), b1=sharedX(b1))
    output = [gx, cost, f_rec, x_rec, x_f, gx_f]

    print 'COMPILING...'
    t = time()

    z_updates = d_updater([z], cost)
    _invert = theano.function(inputs=[x, x_f], outputs=output, updates=z_updates)

    print '%.2f seconds to compile _invert function' % (time() - t)
    return [_invert, z_updates, z]


def def_feature(layer='conv4', up_scale=4):
    x = T.tensor4()
    if layer is 'hog':
        x_f = HOGNet.get_hog(x, use_bin=True, BS=4)
    else:
        x_t = alexnet.transform_im(x)
        x_net = alexnet.build_model(x_t, layer=layer, shape=(None, 3, 64, 64), up_scale=up_scale)
        alexnet.load_model(x_net, layer=layer)
        x_f = lasagne.layers.get_output(x_net[layer], deterministic=True)
    _ftr = theano.function(inputs=[x], outputs=x_f)
    return _ftr


def invert_image_basic(invert_model, ftr_model, ims, npx, z0=None, n_iters=200, isMultiple=False):
    # st()
    [_invert, z_updates, z] = invert_model
    batch_size = ims.shape[0]
    if z0 is None:
        # print('random init')
        z0_f = floatX(np_rng.uniform(-1.0, 1.0, size=(batch_size, nz)))
    else:
        # print('copy %d z0' % z0.shape[0])
        assert z0.shape[0] == ims.shape[0]
        z0_f = floatX(z0)

    lib.modeldef.reset_adam(z_updates)
    z.set_value(floatX(np.arctanh(z0_f)))
    ims_t = lib.modeldef.transform(ims)
    ftrs = ftr_model(ims_t)

    for k in range(n_iters):
        [gx, rec, f_rec, x_rec, x_f, gx_f] = _invert(ims_t, ftrs)
        cost_all = f_rec + x_rec
        id = np.argmin(cost_all)

        print('iter = %3d, total = %3.3f, rec = %3.3f, ftr = %3.3f, pixel =%3.3f' % (
            k, rec, cost_all[id], f_rec[id], x_rec[id]))
    # sys.stdout.flush()
    # st()
    gx = lib.modeldef.inverse_transform(gx, npx=npx)
    zs = np.tanh(z.get_value())
    if isMultiple:
        # st()
        # cost_all = f_rec+x_rec
        order = np.argsort(cost_all)
        cost_all = cost_all[order]
        gx = gx[order]

        zs = zs[order]
    return gx, zs, cost_all


def invert_image_multiple(invert_model, ftr_model, ims, batch_size=1, z0=None, n_iters=1, npx=64):
    zs = []
    recs = []
    fs = []
    t = time()
    for n in range(ims.shape[0]):
        im = ims[[n]]

        # st()
        im_rep = np.tile(im, [batch_size, 1, 1, 1])
        if z0 is not None:
            z0_n = z0[[n]]
            z0_rep = np.tile(z0_n, [batch_size, 1])
        else:
            z0_rep = None
        # z0_rep = np.tile()
        gx, z_value, cost = invert_image_basic(invert_model, ftr_model, ims=im_rep, z0=z0_rep, npx=npx, n_iters=n_iters,
                                               isMultiple=True)

        rec_im = (gx[0] * 255).astype(np.uint8)
        f = cost[0]
        fs.append(f)
        zs.append(z_value[0])
        recs.append(rec_im)
        # st()
        # st()
    print('batch_size = %d, time = %3.3f' % (batch_size, time() - t))
    return recs, zs, fs


def invert_image_batch(invert_model, ftr_model, ims, z0=None, batch_size=32, n_iters=1, npx=64):
    # st()
    zs = []
    recs = []
    fs = []
    t = time()
    n_imgs = ims.shape[0]
    n_batch = int(np.floor(n_imgs / float(batch_size)))
    print('n_batch=%d, batch_size=%d' % (n_batch, batch_size))

    for n in range(n_batch):
        ims_n = ims[n * batch_size:(n + 1) * batch_size, :, :, :]
        if z0 is not None:
            z0_n = z0[n * batch_size:(n + 1) * batch_size, :]
        else:
            z0_n = None

        gx, z_value, cost = invert_image_basic(invert_model, ftr_model, ims=ims_n, z0=z0_n, npx=npx, n_iters=n_iters,
                                               isMultiple=False)
        rec_im = (gx * 255).astype(np.uint8)
        f = cost
        fs.append(f)
        zs.append(z_value)
        recs.append(rec_im)

    n_left = n_imgs - n_batch * batch_size
    # st()
    if n_left > 0:
        if z0 is not None:
            z0_r = floatX(np_rng.uniform(-1.0, 1.0, size=(batch_size - n_left, nz)))
            z0_t = z0[n_batch * batch_size:, :]
            z0_n = np.concatenate([z0_t, z0_r])
        else:
            z0_n = None
        ims_t = ims[n_batch * batch_size:, :, :, :]
        ims_r = ims[0:batch_size - n_left, :, :, :]
        ims_n = np.concatenate([ims_t, ims_r])
        # print(ims_n.shape)
        # print(z0_n.shape)
        # st()
        gx, z_value, cost = invert_image_basic(invert_model, ftr_model, ims=ims_n, z0=z0_n, npx=npx, n_iters=n_iters,
                                               isMultiple=False)
        rec_im = (gx[0:n_left, ...] * 255).astype(np.uint8)
        f = cost[0:n_left]
        z_value = z_value[0:n_left, ...]
        # st()
        fs.append(f)
        zs.append(z_value)
        recs.append(rec_im)

    recs = np.concatenate(recs, axis=0)
    zs = np.concatenate(zs, axis=0)
    fs = np.concatenate(fs, axis=0)
    print('batch_size = %d, time = %3.3f' % (batch_size, time() - t))
    return recs, zs, fs



def invert_bfgs_batch(invert_model, ftr_model, ims, z_predict=None, npx=64):
    # st()
    zs = []
    recs = []
    fs = []
    t = time()
    n_imgs = ims.shape[0]
    print('rec bfgs %d images'%n_imgs)

    for n in range(n_imgs):
        im_n = ims[[n], :, :,:]
        if z_predict is not None:
            z0_n = z_predict[[n],...]
        else:
            z0_n = None
        # st()
        gx, z_value, f_value = invert_bfgs(invert_model, ftr_model,im=im_n, z_predict=z0_n, npx=npx)
        # st()
        rec_im = (gx * 255).astype(np.uint8)
        # f = cost
        # st()
        fs.append(f_value[np.newaxis,...])
        zs.append(z_value[np.newaxis,...])
        recs.append(rec_im)
    # st()
    recs = np.concatenate(recs, axis=0)
    zs = np.concatenate(zs, axis=0)
    fs = np.concatenate(fs, axis=0)
    print('batch_size = %d, time = %3.3f' % (batch_size, time() - t))
    return recs, zs, fs

def invert_bfgs(invert_model, ftr_model, im, z_predict=None, npx=64):
    _f, z = invert_model
    print('invert using bfgs')
    if z_predict is None:
        z_predict = np_rng.uniform(-1., 1., size=(1, nz))
    else:
        z_predict = floatX(z_predict)
    # st()
    z_predict = np.arctanh(z_predict)
    # z.set_value(floatX(np.arctanh(z_predict)))
    im_t = lib.modeldef.transform(im)
    ftr = ftr_model(im_t)

    prob = optimize.minimize(f_bfgs, z_predict, args=(_f, im_t, ftr),
                             tol=1e-6, jac=True, method='L-BFGS-B', options={'maxiter':200})
    print('n_iters = %3d, f = %.3f' % (prob.nit, prob.fun))
    z_opt = prob.x
    z_opt_n = floatX(z_opt[np.newaxis, :])
    # if mask is not None:
    # st()
    [f_opt,  g, gx] = _f(z_opt_n, im_t, ftr)
    # st()
    # st()
    print('cost=%3.3f' % f_opt)
    gx = lib.modeldef.inverse_transform(gx, npx=npx)
    # st()
    z_opt = np.tanh(z_opt)
    # st()
    return gx, z_opt,f_opt


def f_bfgs(z0, _f, x, x_f):
    z0_n = floatX(z0[np.newaxis, :])
    [f, g, gx] = _f(z0_n, x, x_f)
    f = f.astype(np.float64)
    g = g[0].astype(np.float64)
    # print('f=%3.3f'%f)
    return f, g


def def_bfgs(model_dir, desc, layer='conv4',alpha=0.002,npx=64):
    npx, n_layers, n_f, c_iter = getattr(lib.config, desc)()
    [disc_params, gen_params, disc_pl, gen_pl] = lib.modeldef.get_params(model_dir, n_layers, n_f, c_iter)

    x_f = T.tensor4()
    x = T.tensor4()
    z = T.matrix()
    # z = sharedX(floatX(np_rng.uniform(-1., 1., size=(batch_size, nz))))
    gx = lib.modeldef.gen_test_tanh(z, gen_params, gen_pl, n_layers=n_layers, n_f=n_f)

    # st()
    if layer is 'hog':
        gx_f = HOGNet.get_hog(gx, use_bin=True, BS=4)
    else:
        gx_t = alexnet.transform_im(gx)
        gx_net = alexnet.build_model(gx_t, layer=layer, shape=(None, 3, npx, npx))
        alexnet.load_model(gx_net, layer=layer)
        gx_f = lasagne.layers.get_output(gx_net[layer], deterministic=True)

    f_rec = T.mean(T.sqr(x_f - gx_f), axis=(1, 2, 3)) * sharedX(alpha)
    x_rec = T.mean(T.sqr(x - gx), axis=(1, 2, 3))
    cost = T.sum(f_rec) + T.sum(x_rec)
    grad = T.grad(cost, z)
    # d_updater = updates.Adam(lr=sharedX(lr), b1=sharedX(b1))
    output = [cost, grad, gx]

    print 'COMPILING...'
    t = time()

    # z_updates = d_updater([z], cost)
    _invert = theano.function(inputs=[z, x, x_f], outputs=output)

    print '%.2f seconds to compile _invert function' % (time() - t)
    return _invert,z
    # return [_invert, z_updates, z]


def rec_scores(scores, psnr=None):
    if psnr is None:
        return ['f: %3.3f' % s for s in np.nditer(scores)]
    else:
        return ['f: %3.3f, psnr: %3.3fDB' % (f,p) for f,p in zip(np.nditer(scores), np.nditer(psnr))]


def def_models(model_dir, solver='lbfgs', batch_size=16, desc='shoes_64', layer='conv4',
            alpha=0.002, lr=0.1, b1=0.5):
    # st()
    if solver is 'lbfgs':
        invert_model = def_bfgs(model_dir=model_dir, desc=desc, layer=layer, alpha=alpha)
    else:
        invert_model = def_invert(batch_size=batch_size, desc=desc, model_dir=model_dir, layer=layer,
                             alpha=alpha, lr=0.1, b1=0.5)
    ftr_model = def_feature(layer=layer)
    rec_model = def_rec(layer=layer, alpha=alpha)
    psnr_model = def_psnr()
    predict_model = def_predict(model_dir=model_dir, desc=desc, layer=layer, nz=100)
    return invert_model, ftr_model, rec_model, psnr_model, predict_model, batch_size


def invert_images_CNN_opt(models, ims, solver='lbfgs',batch_size=32, n_iters=200, npx=64, type='cnn', gen_model=None):
    invert_model, ftr_model, rec_model, psnr_model, predict_model, batch_size = models
    if type is 'cnn' or type is 'cnn_opt':
        z_predict = predict_z(predict_model, ims, batch_size=batch_size)
    else:
        z_predict = None

    recs = None
    if type is 'cnn':
        assert gen_model is not None
        recs, scores = gen_samples.gen_samples(gen_model, npx=npx, z0=z_predict, n=len(z_predict),
                                               batch_size=batch_size)
        zs = None

    # st()
    if type is 'cnn_opt' or type is 'opt':
        if solver is 'lbfgs':
            recs, zs, loss = invert_bfgs_batch(invert_model, ftr_model, ims, z_predict=z_predict, npx=npx)
        else:
            recs, zs, loss = invert_image_batch(invert_model, ftr_model, ims,
                                            batch_size=batch_size, z0=z_predict, n_iters=n_iters, npx=npx)
    loss, psnr = rec_loss(ims, recs, rec_model, psnr_model)
    # st()
    return recs, loss, psnr, zs, z_predict


def flat_list(list):
    return [item for sublist in list for item in sublist]


import cv2


def init_images(models, impath, desc='shoes_64', im_o=None):
    npx, n_layers, n_f, c_iter = getattr(lib.config, desc)()
    im_name = impath.split('/')[-1]
    cache_dir = '../../cache/%s/' % desc
    # model_dir = '../../models/%s' %desc
    # st()
    utils.mkdirs(cache_dir)
    cache_file = os.path.join(cache_dir, im_name.replace('.png', ''))

    if not os.path.exists(cache_file):
        if im_o is None:
            im_o = cv2.imread(impath)
        im_c = cv2.cvtColor(im_o, cv2.COLOR_BGR2RGB)
        im_s = cv2.resize(im_c, (npx, npx))

        imb = im_s[np.newaxis, :, :, :]
        [rec, loss_rec, loss_psnr, z, z_predict] = invert_images_CNN_opt(models, imb, batch_size=1, n_iters=200,
                                                              npx=npx, type='cnn_opt')
        with open(cache_file, 'wb') as f:
            cPickle.dump([im_o, rec, z], f, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        with open(cache_file, 'rb') as f:
            [im_o, rec, z] = cPickle.load(f)
            im_c = cv2.cvtColor(im_o, cv2.COLOR_BGR2RGB)
    return im_c, z, rec


if __name__ == "__main__":
    desc = 'shoes_64'
    # c_iter = 40  # handbag, shirts: 40, shoes,churches:50
    w = 128
    layer = 'conv4'
    expr = 'test'
    db_name ='amazon_handbag'
    # alpha = 0
    # if layer is 'hog':
    db_name = 'amazon_shirts'
    alpha = 0.002
    ext=''
    expr_name = desc +ext

    data_dir = '../../dataset/%s/images.hdf5' % db_name

    model_dir = '../../models/%s' % expr_name
    npx, n_layers, n_f, c_iter = getattr(lib.config, desc)()
    #home_dir = '/home/eecs/junyanz/public_html/projects/gvm/'
    home_dir='../../web/'
    if expr is 'feature':
        batch_size = 128
        import scipy.io

        ftr_matlab = '../../dataset/%s/ftrs_train_%s/' % (desc, layer)
        utils.mkdirs(ftr_matlab)
        # if not os.path.exists(ftr_matlab):
        #     os.makedirs(ftr_matlab)
        feature_model = def_feature(layer=layer, up_scale=2)
        tr_data, te_data, tr_stream, te_stream, ntrain, ntest = load_imgs_seq(ntrain=None, ntest=None,
                                                                              batch_size=batch_size, data_dir=data_dir)
        alpha2 = np.sqrt(alpha)
        # ftrs = []
        batch_id = 0
        for imb, in tqdm(tr_stream.get_epoch_iterator(), total=ntrain / batch_size):
            imb = lib.modeldef.transform(imb)
            # st()
            batch_n = len(imb)
            # st()
            conv_ftr = feature_model(imb) * alpha2
            conv_ftr = np.reshape(conv_ftr, [batch_n, -1])
            pixel_ftr = np.reshape(imb, [batch_n, -1])
            # st()
            # st()
            ftr = np.concatenate([conv_ftr, pixel_ftr], axis=1)
            # print(ftr)
            # st()
            scipy.io.savemat(os.path.join(ftr_matlab, 'batch_%5.5d.mat' % batch_id), mdict={'ftr': ftr})
            batch_id += 1

    if expr is 'test':
        batch_size = 50
        n_batches = 10
        tr_data, te_data, tr_stream, te_stream, ntrain, ntest = load_imgs(ntrain=None, ntest=batch_size * n_batches,
                                                                          batch_size=batch_size, data_dir=data_dir)
        z_dir = '../../results/%s_%s/bfgs_test_%s/' % (expr_name, db_name, layer)
        test_result = os.path.join(z_dir, 'tmp_result')
        # methods = ['opt']
        solver = 'lbfgs'
        web_name = 'rec_lbfgs/add_%s_iter%d_%s_alpha%3.3f' % (desc, c_iter, layer, alpha)

        methods = [ 'opt','cnn', 'cnn_opt']
        utils.mkdirs(z_dir)
        if os.path.exists(test_result):
            with open(test_result, 'rb') as f:
                [loss_all,psnr_all]= cPickle.load(f)


        else:
            n_total = 0
            gen_model, gen_params = gen_samples.def_gen_score(model_dir, n_layers, n_f, c_iter)
            models = def_models(batch_size=batch_size, solver=solver, desc=desc, model_dir=model_dir, layer=layer,
                              alpha=alpha, lr=0.1, b1=0.9)

            batch_id = 0

            web_dir = os.path.join(home_dir, web_name)
            html = image_save.ImageSave(web_dir, web_name, append=True)
            all_zs = []
            all_pred = []

            loss_all = [[] for method in methods]
            psnr_all = [[] for method in methods]

            for imb, in tqdm(te_stream.get_epoch_iterator(), total=ntest / batch_size):
                # z_file = os.path.join(z_dir, 'batch%5.5d_num%d' % (batch_id, batch_size))
                # if not os.path.exists(z_file):
                txts = [''] * imb.shape[0]
                html.save_image(imb, txts, header='original images', width=w, cvt=True)
                html.save()

                for m, method in enumerate(methods):
                    [recs, loss, psnr, zs, z_predict] = invert_images_CNN_opt(models, imb, solver=solver,batch_size=batch_size, n_iters=200,
                                                                        npx=npx, type=method, gen_model=gen_model)
                    if batch_id % 1 == 0:
                        txt_loss = rec_scores(loss, psnr)
                        loss_mean = np.mean(loss)
                        psnr_mean = np.mean(psnr)
                        html.save_image(recs, txt_loss,
                                        header='%s method, batch_id %d, rec: %3.3f, psnr = %3.3fDB'
                                               % (method, batch_id, loss_mean, psnr_mean),width=w, cvt=True)
                        html.save()
                    # st()
                    loss_all[m] += loss
                    psnr_all[m] += [psnr]

                batch_id += 1
                if batch_id == n_batches:
                    break
            # with open(test_result, 'wb') as f:
            #     cPickle.dump([loss_all,psnr_all], f, protocol=cPickle.HIGHEST_PROTOCOL)
        print('dataset: %s, expr_name: %s'%(db_name, expr_name))
        for m, method in enumerate(methods):
            flat_loss = flat_list(loss_all[m])
            flat_psnr = flat_list(psnr_all[m])
            txt = 'method %s: f %.3f, psnr=%3.3fdB' % (method, np.mean(flat_loss),np.mean(flat_psnr))
            print(txt)



            # all_zs = np.concatenate(all_zs, axis=0)
            # all_pred = np.concatenate(all_pred, axis=0)
            # z_python ='../../dataset/%s/z'%desc
            # z_matlab ='../../dataset/%s/z.mat'%desc

            # import scipy.io
            # scipy.io.savemat(z_matlab, mdict={'z': all_pred})

    if expr is 'predict':
        batch_size = 128
        tr_data, te_data, tr_stream, te_stream, ntrain, ntest = load_imgs(ntrain=None, ntest=None,
                                                                          batch_size=batch_size, data_dir=data_dir)
        z_dir = '../../results/%s/z_save_%s/' % (desc, layer)
        utils.mkdirs(z_dir)
        n_total = 0
        models = def_models(batch_size=batch_size, model_dir=model_dir, n_layers=n_layers, layer=layer, c_iter=c_iter,
                            n_f=n_f, alpha=alpha, lr=0.1, b1=0.5)
        batch_id = 0
        web_name = 'train_rec/%s_iter%d_%s_alpha%3.3f' % (desc, c_iter, layer, alpha)
        web_dir = os.path.join(home_dir, web_name)
        html = image_save.ImageSave(web_dir, web_name, append=True)
        all_zs = []
        all_pred = []

        for imb, in tqdm(tr_stream.get_epoch_iterator(), total=ntrain / batch_size):
            z_file = os.path.join(z_dir, 'batch%5.5d_num%d' % (batch_id, batch_size))
            if not os.path.exists(z_file):
                [recs, loss_rec, zs, z_predict] = invert_images_CNN_opt(models, imb, batch_size=batch_size, n_iters=200,
                                                                        npx=npx, type='cnn_opt')
                all_zs.append(zs)
                all_pred.append(z_predict)
                with open(z_file, 'wb') as f:
                    cPickle.dump([loss_rec, zs, z_predict], f, protocol=cPickle.HIGHEST_PROTOCOL)

                if batch_id % 50 == 0:
                    txts = rec_scores(loss_rec)
                    loss_mean = np.mean(loss_rec)
                    txts = [''] * imb.shape[0]
                    html.save_image(imb, txts, header='original images', width=w, cvt=True)
                    html.save_image(recs, txts, header='CNN+OPT %d,  loss: %3.3f' % (batch_id, loss_mean), width=w,
                                    cvt=True)
                    html.save()
            else:
                with open(z_file, 'rb') as f:
                    [loss_rec, zs, z_predict] = cPickle.load(f)
                    all_zs.append(zs)
                    all_pred.append(z_predict)
                    # st()
            batch_id += 1

        all_zs = np.concatenate(all_zs, axis=0)
        all_pred = np.concatenate(all_pred, axis=0)
        z_python = '../../dataset/%s/z' % desc
        z_matlab = '../../dataset/%s/z.mat' % desc
        with open(z_python, 'wb') as f:
            cPickle.dump(all_pred, f, protocol=cPickle.HIGHEST_PROTOCOL)
        import scipy.io

        scipy.io.savemat(z_matlab, mdict={'z': all_pred})

    if expr is 'compare':
        web_name = 'rec/rec_%s_iter%d_%s_alpha%3.3f_comp_s2' % (desc, c_iter, layer, alpha)
        web_dir = os.path.join(home_dir, web_name)

        tr_data, te_data = load_imgs_raw(data_dir=data_dir)
        tr_handle = tr_data.open()
        batch_size = 1
        ims = tr_data.get_data(tr_handle, slice(0, 15))
        ims = ims[0]

        html = image_save.ImageSave(web_dir, web_name, append=True)
        # save original images
        txts = [''] * ims.shape[0]
        html.save_image(ims, txts, header='original images', width=w, cvt=True)
        models = def_models(batch_size=batch_size, model_dir=model_dir, n_layers=n_layers, layer=layer, c_iter=c_iter,
                            n_f=n_f, alpha=alpha, lr=0.1, b1=0.5)

        [recs, loss_rec, zs, z_predict] = invert_images_CNN_opt(models, ims, n_iters=200, npx=npx, type='opt')
        txts = rec_scores(loss_rec)
        loss_mean = np.mean(loss_rec)
        html.save_image(recs, txts, header='<Optimization> mean loss: %3.3f' % loss_mean, width=w, cvt=True)

        gen_model, gen_params = gen_samples.def_gen_score(model_dir, n_layers, n_f, c_iter)
        [recs, loss_rec, zs, z_predict] = invert_images_CNN_opt(models, ims, n_iters=200, npx=npx, type='cnn',
                                                                gen_model=gen_model)
        txts = rec_scores(loss_rec)
        loss_mean = np.mean(loss_rec)
        html.save_image(recs, txts, header='<CNN> mean loss: %3.3f' % loss_mean, width=w, cvt=True)

        [recs, loss_rec, zs, z_predict] = invert_images_CNN_opt(models, ims, n_iters=200, npx=npx, type='cnn_opt')
        txts = rec_scores(loss_rec)
        loss_mean = np.mean(loss_rec)
        html.save_image(recs, txts, header='<CNN + Optimization> mean loss: %3.3f' % loss_mean, width=w, cvt=True)
        html.save()
