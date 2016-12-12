import theano
import theano.tensor as T
from time import time
from lib import updates, HOGNet
from lib.rng import np_rng
from lib.theano_utils import floatX, sharedX
import numpy as np

class OPT_Solver():
    def __init__(self, model, batch_size=32, d_weight=0.0):
        self.model = model
        self.npx = model.npx
        self.nc = model.nc
        self.nz = model.nz
        self.model_name = model.model_name
        self.transform = model.transform
        self.transform_mask = model.transform_mask
        self.inverse_transform = model.inverse_transform
        BS = 4 if self.nc == 1 else 8 # [hack]
        self.hog = HOGNet.HOGNet(use_bin=True, NO=8, BS=BS, nc=self.nc)
        self.opt_model = self.def_invert(model, batch_size=batch_size, d_weight=d_weight, nc=self.nc)
        self.batch_size = batch_size

    def get_image_size(self):
        return self.npx

    def invert(self, constraints, z_i):
        [_invert, z_updates, z, beta_r, z_const] = self.opt_model
        constraints_t  = self.preprocess_constraints(constraints)
        [im_c_t, mask_c_t, im_e_t, mask_e_t] = constraints_t # [im_c_t, mask_c_t, im_e_t, mask_e_t]

        results = _invert(im_c_t, mask_c_t, im_e_t, mask_e_t, z_i.astype(np.float32))

        [gx, cost, cost_all, rec_all, real_all, init_all, sum_e, sum_x_edge] = results
        gx_t = (255 * self.inverse_transform(gx, npx=self.npx, nc=self.nc)).astype(np.uint8)
        if self.nc == 1:
            gx_t = np.tile(gx_t, (1, 1, 1, 3))
        z_t = np.tanh(z.get_value()).copy()
        return gx_t, z_t, cost_all


    def preprocess_constraints(self, constraints):
        [im_c_o, mask_c_o, im_e_o, mask_e_o] = constraints
        im_c = self.transform(im_c_o[np.newaxis, :], self.nc)
        mask_c = self.transform_mask(mask_c_o[np.newaxis, :])
        im_e = self.transform(im_e_o[np.newaxis, :], self.nc)
        mask_t = self.transform_mask(mask_e_o[np.newaxis, :])
        mask_e = self.hog.comp_mask(mask_t)
        shp = [self.batch_size, 1, 1, 1]
        im_c_t = np.tile(im_c, shp)
        mask_c_t = np.tile(mask_c, shp)
        im_e_t = np.tile(im_e, shp)
        mask_e_t = np.tile(mask_e, shp)
        return [im_c_t, mask_c_t, im_e_t, mask_e_t]

    def initialize(self, z0):
        z = self.opt_model[2]
        z.set_value(floatX(np.arctanh(z0)))

    def set_smoothness(self, l):
        print('set z const = 0')
        z_const = self.opt_model[-1]
        z_const.set_value(floatX(l))

    def gen_samples(self, z0):
        samples = self.model.gen_samples(z0=z0)
        if self.nc == 1:
            samples = np.tile(samples, [1,1,1,3])
        return samples

    def def_invert(self, model, batch_size=1, d_weight=0.5, nc=1, lr=0.1, b1=0.9, nz=100, use_bin=True):
        d_weight_r = sharedX(d_weight)
        x_c = T.tensor4()
        m_c = T.tensor4()
        x_e = T.tensor4()
        m_e = T.tensor4()
        z0 = T.matrix()
        z = sharedX(floatX(np_rng.uniform(-1., 1., size=(batch_size, nz))))
        gx = model.model_G(z)
        # input: im_c: 255: no edge; 0: edge; transform=> 1: no edge, 0: edge

        if nc == 1: # gx, range [0, 1] => edge, 1
            gx3 = 1.0-gx #T.tile(gx, (1, 3, 1, 1))
        else:
            gx3 = gx
        mm_c = T.tile(m_c, (1, gx3.shape[1], 1, 1))
        color_all = T.mean(T.sqr(gx3 - x_c) * mm_c, axis=(1, 2, 3)) / (T.mean(m_c, axis=(1, 2, 3)) + sharedX(1e-5))
        gx_edge = self.hog.get_hog(gx3)
        x_edge = self.hog.get_hog(x_e)
        mm_e = T.tile(m_e, (1, gx_edge.shape[1], 1, 1))
        sum_e = T.sum(T.abs_(mm_e))
        sum_x_edge = T.sum(T.abs_(x_edge))
        edge_all = T.mean(T.sqr(x_edge - gx_edge) * mm_e, axis=(1, 2, 3)) / (T.mean(m_e, axis=(1, 2, 3)) + sharedX(1e-5))
        rec_all = color_all + edge_all * sharedX(0.2)
        z_const = sharedX(5.0)
        init_all = T.mean(T.sqr(z0 - z)) * z_const

        if d_weight > 0:
            print('using D')
            p_gen = model.model_D(gx)
            real_all = T.nnet.binary_crossentropy(p_gen, T.ones(p_gen.shape)).T
            cost_all = rec_all + d_weight_r * real_all[0] + init_all
        else:
            print('without D')
            cost_all = rec_all + init_all
            real_all = T.zeros(cost_all.shape)

        cost = T.sum(cost_all)
        d_updater = updates.Adam(lr=sharedX(lr), b1=sharedX(b1))
        output = [gx, cost, cost_all, rec_all, real_all, init_all, sum_e, sum_x_edge]

        print('COMPILING...')
        t = time()

        z_updates = d_updater([z], cost)
        _invert = theano.function(inputs=[x_c, m_c, x_e, m_e, z0], outputs=output, updates=z_updates)
        print('%.2f seconds to compile _invert function' % (time() - t))
        return [_invert, z_updates, z, d_weight_r, z_const]

