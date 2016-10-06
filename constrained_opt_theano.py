import theano
import theano.tensor as T
from time import time
from lib import updates, HOGNet
from lib.rng import np_rng
from lib.theano_utils import floatX, sharedX
import numpy as np
from lib import utils
from PyQt4.QtCore import *


class Constrained_OPT(QThread):
    def __init__(self, model, batch_size=32, n_iters=25, topK=16, morph_steps=16, nc=3, interp='linear'):
        QThread.__init__(self)
        self.model = model
        self.npx = model.npx
        self.nc = model.nc
        self.model_name = model.model_name
        self.transform = model.transform
        self.transform_mask = model.transform_mask
        self.inverse_transform = model.inverse_transform
        self.invert_model = self.def_invert(batch_size=batch_size, model=self.model, beta=0)
        # data
        self.z_seq = None
        self.img_seq = None
        self.im0 = None
        self.z0 = None
        self.topK = topK
        self.num_frames = n_iters
        self.max_iters = n_iters
        self.batch_size = batch_size
        # constraints
        self.constraints = None
        self.constraints_t = None
        # current frames
        self.current_ims = None
        self.iter_count = 0
        self.iter_total = 0
        self.to_update = False
        self.to_set_constraints = False
        self.order = None


        self.prev_z = self.z0
        self.init_constraints()
        self.init_z()
        self.morph_steps = morph_steps
        self.interp = interp
        self.just_fixed = True

    def morph_between_images(self):
        self.current_zs = self.z_target
         # z1 = self.z1[self.order]
        self.z1 = self.z0
        self.current_ims=self.im_target
        self.order = [0]
        self.gen_morphing(self.model, self.interp, self.morph_steps)


    def isInited(self):
        return self.iter_total == 0

    def set_z(self, frame_id, image_id):
        if self.z_seq is not None:
            self.prev_z = self.z_seq[image_id, frame_id]

    def init_z(self, frame_id=0, image_id=0):
        # print('init z!!!!!')
        nz = 100
        n_sigma = 0.5
        self.iter_total = 0

        # set prev_z
        if self.z_seq is not None:
            image_id = image_id % self.z_seq.shape[0]
            frame_id = frame_id % self.z_seq.shape[1]
            print('set z as image %d, frame %d' % (image_id, frame_id))
            self.prev_z = self.z_seq[image_id, frame_id]

        if self.prev_z is None:  #random initialization
            self.z0_f = floatX(np_rng.uniform(-1.0, 1.0, size=(self.batch_size, nz)))
            self.zero_z_const()
            self.z_i = self.z0_f.copy()
            self.z1 = self.z0_f.copy()
        else:
            z0_r = np.tile(self.prev_z, [self.batch_size, 1])
            z0_n = floatX(np_rng.uniform(-1.0, 1.0, size=(self.batch_size, nz)) * n_sigma)
            self.z0_f = floatX(np.clip(z0_r + z0_n, -0.99, 0.99))
            self.z_i = np.tile(self.prev_z, [self.batch_size, 1])
            self.z1 = z0_r.copy()

        z = self.invert_model[2]
        z.set_value(floatX(np.arctanh(self.z0_f)))
        self.just_fixed = True

    def update(self):   # update ui
        self.to_update = True
        self.to_set_constraints = True
        self.iter_count = 0
        self.img_seq = None

    def save_constraints(self):
        [im_c, mask_c, im_e, mask_e] = self.combine_constraints(self.constraints)
        self.prev_im_c = im_c.copy()
        self.prev_mask_c = mask_c.copy()
        self.prev_im_e = im_e.copy()
        self.prev_mask_e =mask_e.copy()

    def init_constraints(self):
        self.prev_im_c = np.zeros((self.npx, self.npx, self.nc), np.uint8)
        self.prev_mask_c = np.zeros((self.npx, self.npx, 1), np.uint8)
        self.prev_im_e = np.zeros((self.npx, self.npx, self.nc), np.uint8)
        self.prev_mask_e = np.zeros((self.npx, self.npx, 1), np.uint8)


    def combine_constraints(self, constraints):
        if constraints is not None:
            print('combine strokes')
            [im_c, mask_c, im_e, mask_e] = constraints
            mask_c_f = np.maximum(self.prev_mask_c, mask_c)
            mask_e_f = np.maximum(self.prev_mask_e, mask_e)

            im_c_f =  self.prev_im_c.copy()
            mask_c3 = np.tile(mask_c, [1,1,3])
            np.copyto(im_c_f, im_c, where=mask_c3.astype(np.bool))  #[hack]
            im_e_f = self.prev_im_e.copy()
            mask_e3 = np.tile(mask_e, [1,1,3])
            np.copyto(im_e_f, im_e, where=mask_e3.astype(np.bool))

            return [im_c_f, mask_c_f, im_e_f, mask_e_f]
        else:
            return [self.prev_im_c, self.prev_mask_c, self.prev_im_e, self.prev_mask_e]

    def preprocess_constraints(self, constraints):
        [im_c_o, mask_c_o, im_e_o, mask_e_o] = self.combine_constraints(constraints)
        im_c = self.transform(im_c_o[np.newaxis, :])
        mask_c= self.transform_mask(mask_c_o[np.newaxis, :])
        im_e = self.transform(im_e_o[np.newaxis, :])
        mask_t = self.transform_mask(mask_e_o[np.newaxis, :])
        mask_e = HOGNet.comp_mask(mask_t)
        shp = [self.batch_size, 1, 1, 1]
        im_c_t = np.tile(im_c, shp)
        mask_c_t = np.tile(mask_c, shp)
        im_e_t = np.tile(im_e, shp)
        mask_e_t = np.tile(mask_e, shp)
        return [im_c_t, mask_c_t, im_e_t, mask_e_t]

    def set_constraints(self, constraints):
        self.constraints = constraints

    def get_z(self, image_id, frame_id):
        if self.z_seq is not None:
            image_id = image_id % self.z_seq.shape[0]
            frame_id = frame_id % self.z_seq.shape[1]
            return self.z_seq[image_id, frame_id]
        else:
            return None

    def run(self): #
        time_to_wait = 33 # 33 millisecond
        while (1):
            t1 =time()
            if self.to_set_constraints:# update constraints
                self.constraints_t = self.preprocess_constraints(self.constraints)
                self.to_set_constraints = False

            if self.constraints_t is not None and self.iter_count < self.max_iters:
                self.update_invert(constraints=self.constraints_t)
                self.iter_count += 1
                self.iter_total += 1

            if self.iter_count == self.max_iters:
                self.gen_morphing(self.model, self.interp, self.morph_steps)
                self.to_update = False
                self.iter_count += 1

            t_c = int(1000*(time()-t1))

            if t_c > 0:
                print('update one iteration: %d ms' % t_c)
            if t_c < time_to_wait:
                self.msleep(time_to_wait-t_c)


    def update_invert(self, constraints):
        [_invert, z_updates, z, beta_r, z_const] = self.invert_model
        [im_c_t, mask_c_t, im_e_t, mask_e_t] = constraints
        t = time()

        results = _invert(im_c_t, mask_c_t, im_e_t, mask_e_t, self.z_i.astype(np.float32))#output = [gx, cost, cost_all, rec_all, real_all, init_all, gx_edge, x_edge]
        [gx, cost, cost_all, rec_all, real_all, init_all, sum_e, sum_x_edge] = results
        gx_t = (255 * self.inverse_transform(gx, npx=self.npx)).astype(np.uint8)
        z_t = np.tanh(z.get_value()).copy()

        rec_mean = np.sum(rec_all)
        real_mean = np.sum(real_all)
        init_sum = np.sum(init_all)
        order = np.argsort(cost_all)

        if self.topK > 1:
            cost_sort = cost_all[order]
            thres_top =  2 * np.mean(cost_sort[0:min(int(self.topK / 2.0), len(cost_sort))])
            ids = cost_sort < thres_top
            topK = np.min([self.topK, sum(ids)])
        else:
            topK = self.topK

        order = order[0:topK]

        if self.iter_total < 150:
            self.order = order
        else:
            order = self.order
        self.current_ims = gx_t[order]
        self.current_zs = z_t[order]

        self.emit(SIGNAL('update_image'))

    def get_image(self, image_id, frame_id):
        if self.to_update:
            if self.current_ims is None or self.current_ims.size == 0:
                return None
            else:
                image_id = image_id % self.current_ims.shape[0]
                return self.current_ims[image_id]
        else:

            if self.img_seq is None:
                return None
            else:
                frame_id = frame_id % self.img_seq.shape[1]
                image_id = image_id % self.img_seq.shape[0]
                return  self.img_seq[image_id, frame_id]

    def get_images(self, frame_id):
        if self.to_update:
            return self.current_ims
        else:
            if self.img_seq is None:
                return None
            else:
                frame_id = frame_id % self.img_seq.shape[1]
                return self.img_seq[:, frame_id]

    def gen_morphing(self, model, interp='linear', n_steps=8):
        if self.current_ims is None:
            return

        z1 = self.z1[self.order]
        z2 = self.current_zs
        t = time()
        img_seq = []
        z_seq = []

        for n in range(n_steps):
            ratio = n / float(n_steps- 1)
            z_t = utils.interp_z(z1, z2, ratio, interp=interp)
            seq = model.gen_samples(z0=z_t)
            img_seq.append(seq[:, np.newaxis, ...])
            z_seq.append(z_t[:,np.newaxis,...])
        self.img_seq = np.concatenate(img_seq, axis=1)
        self.z_seq = np.concatenate(z_seq, axis=1)
        print('generate morphing sequence (%.3f seconds)' % (time()-t))

    def reset(self):
        self.z_seq = None
        self.img_seq = None
        self.prev_z = self.z0  # .copy()
        self.constraints = None
        self.constraints_t = None
        self.current_ims = None
        self.iter_count = 0
        self.to_update = False
        self.order = None
        self.to_set_constraints = False
        self.iter_total = 0
        self.init_z()
        self.init_constraints()

    def zero_z_const(self):
        print('set z const = 0')
        z_const = self.invert_model[-1]
        z_const.set_value(floatX(0))

    def get_num_images(self):
        if self.img_seq is None:
            return 0
        else:
            return self.img_seq.shape[0]

    def get_num_frames(self):
        if self.img_seq is None:
            return 0
        else:
            return self.img_seq.shape[1]


    def def_invert(self, model, batch_size=1, beta=0.5, lr=0.1, b1=0.9, nz=100, use_bin=True):
        beta_r = sharedX(beta)
        x_c = T.tensor4()
        m_c = T.tensor4()
        x_e = T.tensor4()
        m_e = T.tensor4()
        z0 = T.matrix()
        z = sharedX(floatX(np_rng.uniform(-1., 1., size=(batch_size, nz))))
        gx = model.model_G(z)

        mm_c = T.tile(m_c, (1, gx.shape[1], 1, 1))
        color_all = T.mean(T.sqr(gx - x_c) * mm_c, axis=(1, 2, 3)) / (T.mean(m_c, axis=(1, 2, 3)) + sharedX(1e-5))
        gx_edge = HOGNet.get_hog(gx, use_bin)
        x_edge = HOGNet.get_hog(x_e, use_bin)
        mm_e = T.tile(m_e, (1, gx_edge.shape[1], 1, 1))
        sum_e = T.sum(T.abs_(mm_e))
        sum_x_edge = T.sum(T.abs_(x_edge))
        edge_all = T.mean(T.sqr(x_edge - gx_edge) * mm_e, axis=(1, 2, 3)) / (T.mean(m_e, axis=(1, 2, 3)) + sharedX(1e-5))
        rec_all = color_all + edge_all * sharedX(0.2)
        z_const = sharedX(10.0)
        init_all = T.mean(T.sqr(z0 - z)) * z_const

        if beta > 0:
            print('using D')
            p_gen = model.model_D(gx)
            real_all = T.nnet.binary_crossentropy(p_gen, T.ones(p_gen.shape)).T
            cost_all = rec_all + beta_r * real_all[0] + init_all
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
        return [_invert, z_updates, z, beta_r, z_const]

