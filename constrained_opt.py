from __future__ import print_function
from time import time
from lib.rng import np_rng
import numpy as np
import sys
from lib import utils
from PyQt4.QtCore import *
import cv2

class Constrained_OPT(QThread):
    def __init__(self, opt_solver, batch_size=32, n_iters=25, topK=16, morph_steps=16, interp='linear'):
        QThread.__init__(self)
        self.nz = 100
        self.opt_solver = opt_solver
        self.topK = topK
        self.max_iters = n_iters
        self.fixed_iters = 150  # [hack] after 150 iterations, do not change the order of the results
        self.batch_size = batch_size
        self.morph_steps = morph_steps  # number of intermediate frames
        self.interp = interp  # interpolation method
        # data
        self.z_seq = None     # sequence of latent vector
        self.img_seq = None   # sequence of images
        self.im0 = None       # initial image
        self.z0 = None        # initial latent vector
        self.prev_z = self.z0  # previous latent vector
        # constraints
        self.constraints = None
        # current frames
        self.current_ims = None   # the images being displayed now
        self.iter_count = 0
        self.iter_total = 0
        self.to_update = False
        self.to_set_constraints = False
        self.order = None
        self.init_constraints()  # initialize
        self.init_z()            # initialize latent vectors
        self.just_fixed=True
        self.weights = None

    def is_fixed(self):
        return self.just_fixed

    def update_fix(self):
        self.just_fixed = False

    def init_z(self, frame_id=-1, image_id=-1):
        nz = self.nz
        n_sigma = 0.5
        self.iter_total = 0
        # set prev_z
        if self.z_seq is not None and image_id >= 0:
            image_id = image_id % self.z_seq.shape[0]
            frame_id = frame_id % self.z_seq.shape[1]
            print('set z as image %d, frame %d' % (image_id, frame_id))
            self.prev_z = self.z_seq[image_id, frame_id]

        if self.prev_z is None:  #random initialization
            self.z_init = np_rng.uniform(-1.0, 1.0, size=(self.batch_size, nz))
            self.opt_solver.set_smoothness(0.0)
            self.z_const = self.z_init
            self.prev_zs = self.z_init
        else:  # add small noise to initial latent vector, so that we can get different results
            z0_r = np.tile(self.prev_z, [self.batch_size, 1])
            z0_n = np_rng.uniform(-1.0, 1.0, size=(self.batch_size, nz)) * n_sigma
            self.z_init = np.clip(z0_r + z0_n, -0.99, 0.99)
            self.opt_solver.set_smoothness(5.0)
            self.z_const = np.tile(self.prev_z, [self.batch_size, 1])
            self.prev_zs = z0_r

        self.opt_solver.initialize(self.z_init)
        self.just_fixed = True

    def update(self):   # update ui
        self.to_update = True
        self.to_set_constraints = True
        self.iter_count = 0
        self.img_seq = None

    def save_constraints(self):
        [im_c, mask_c, im_e, mask_e] = self.combine_constraints(self.constraints)
        # write image
        # im_c2 = cv2.cvtColor(im_c, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('input_color_image.png', im_c2)
        # cv2.imwrite('input_color_mask.png', mask_c)
        # cv2.imwrite('input_edge_map.png', im_e)
        self.prev_im_c = im_c.copy()
        self.prev_mask_c = mask_c.copy()
        self.prev_im_e = im_e.copy()
        self.prev_mask_e =mask_e.copy()

    def init_constraints(self):
        self.prev_im_c = None
        self.prev_mask_c = None
        self.prev_im_e = None
        self.prev_mask_e = None


    def combine_constraints(self, constraints):
        if constraints is not None: #[hack]
            # print('combine strokes')
            [im_c, mask_c, im_e, mask_e] = constraints
            if self.prev_im_c is None:
                mask_c_f = mask_c
            else:
                mask_c_f = np.maximum(self.prev_mask_c, mask_c)

            if self.prev_im_e is None:
                mask_e_f = mask_e
            else:
                mask_e_f = np.maximum(self.prev_mask_e, mask_e)

            if self.prev_im_c is None:
                im_c_f = im_c
            else:
                im_c_f =  self.prev_im_c.copy()
                mask_c3 = np.tile(mask_c, [1,1, im_c.shape[2]])
                np.copyto(im_c_f, im_c, where=mask_c3.astype(np.bool))  #[hack]

            if self.prev_im_e is None:
                im_e_f = im_e
            else:
                im_e_f = self.prev_im_e.copy()
                mask_e3 = np.tile(mask_e, [1,1,im_e.shape[2]])
                np.copyto(im_e_f, im_e, where=mask_e3.astype(np.bool))

            return [im_c_f, mask_c_f, im_e_f, mask_e_f]
        else:
            return [self.prev_im_c, self.prev_mask_c, self.prev_im_e, self.prev_mask_e]


    def set_constraints(self, constraints):
        self.constraints = constraints

    def get_z(self, image_id, frame_id):
        if self.z_seq is not None:
            image_id = image_id % self.z_seq.shape[0]
            frame_id = frame_id % self.z_seq.shape[1]
            return self.z_seq[image_id, frame_id]
        else:
            return None


    def get_image(self, image_id, frame_id, useAverage=False):
        if self.to_update:
            if self.current_ims is None or self.current_ims.size == 0:
                return None
            else:
                image_id = image_id % self.current_ims.shape[0]
                if useAverage and self.weights is not None:
                    return utils.average_image(self.current_ims, self.weights)  # get averages
                else:
                    return self.current_ims[image_id]
        else:
            if self.img_seq is None:
                return None
            else:
                frame_id = frame_id % self.img_seq.shape[1]
                image_id = image_id % self.img_seq.shape[0]
                if useAverage and self.weights is not None:
                    return utils.average_image(self.img_seq[:,frame_id,...], self.weights)
                else:
                    return self.img_seq[image_id, frame_id]

    def get_images(self, frame_id):
        if self.to_update:
            return self.current_ims
        else:
            if self.img_seq is None:
                return None
            else:
                frame_id = frame_id % self.img_seq.shape[1]
                return self.img_seq[:, frame_id]

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

    def get_current_results(self):
        return self.current_ims

    def run(self):  # main function
        time_to_wait = 33 # 33 millisecond
        while (1):
            t1 =time()
            if self.to_set_constraints:# update constraints
                self.to_set_constraints = False

            if self.constraints is not None and self.iter_count < self.max_iters:
                self.update_invert(constraints=self.constraints)
                self.iter_count += 1
                self.iter_total += 1

            if self.iter_count == self.max_iters:
                self.gen_morphing(self.interp, self.morph_steps)
                self.to_update = False
                self.iter_count += 1

            t_c = int(1000*(time()-t1))
            print('update one iteration: %03d ms' % t_c, end='\r')
            sys.stdout.flush()
            if t_c < time_to_wait:
                self.msleep(time_to_wait-t_c)

    def update_invert(self, constraints):
        constraints_c = self.combine_constraints(constraints)
        gx_t, z_t, cost_all = self.opt_solver.invert(constraints_c, self.z_const)

        order = np.argsort(cost_all)

        if self.topK > 1:
            cost_sort = cost_all[order]
            thres_top =  2 * np.mean(cost_sort[0:min(int(self.topK / 2.0), len(cost_sort))])
            ids = cost_sort - thres_top < 1e-10
            topK = np.min([self.topK, sum(ids)])
        else:
            topK = self.topK

        order = order[0:topK]

        if self.iter_total < self.fixed_iters:
            self.order = order
        else:
            order = self.order
        self.current_ims = gx_t[order]
        # compute weights
        cost_weights = cost_all[order]
        self.weights = np.exp(-(cost_weights-np.mean(cost_weights)) / (np.std(cost_weights)+1e-10))
        self.current_zs = z_t[order]
        self.emit(SIGNAL('update_image'))

    def gen_morphing(self, interp='linear', n_steps=8):
        if self.current_ims is None:
            return

        z1 = self.prev_zs[self.order]
        z2 = self.current_zs
        t = time()
        img_seq = []
        z_seq = []

        for n in range(n_steps):
            ratio = n / float(n_steps- 1)
            z_t = utils.interp_z(z1, z2, ratio, interp=interp)
            seq = self.opt_solver.gen_samples(z0=z_t)
            img_seq.append(seq[:, np.newaxis, ...])
            z_seq.append(z_t[:,np.newaxis,...])
        self.img_seq = np.concatenate(img_seq, axis=1)
        self.z_seq = np.concatenate(z_seq, axis=1)
        print('generate morphing sequence (%.3f seconds)' % (time()-t))

    def reset(self):
        self.prev_z = self.z0
        self.init_z()
        self.init_constraints()
        self.just_fixed = True
        self.z_seq = None
        self.img_seq = None
        self.constraints = None
        self.current_ims = None
        self.to_update = False
        self.order = None
        self.to_set_constraints = False
        self.iter_total = 0
        self.iter_count = 0
        self.weights =None



