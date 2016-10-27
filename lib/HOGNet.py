import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
from theano_utils import floatX, sharedX
from time import time
# NO = 8
# BS = 8

class HOGNet():
    def __init__(self, use_bin=True, NO=8, BS=8, nc=3):
        self.use_bin=True
        self.NO = NO
        self.BS = BS
        self.nc = nc
        self.use_bin = use_bin
        self._comp_mask = self.def_comp_mask()


    def def_comp_mask(self):
        BS = self.BS
        print('COMPILING')
        t = time()
        m = T.tensor4()
        bf_w = np.ones((1, 1, 2 * BS, 2 * BS))
        bf = sharedX(floatX(bf_w))
        m_b = dnn_conv(m, bf, subsample=(BS, BS), border_mode=(BS / 2, BS / 2))
        _comp_mask = theano.function(inputs=[m], outputs=m_b)
        print('%.2f seconds to compile [compMask] functions' % (time() - t))
        return _comp_mask

    def comp_mask(self, masks):
        masks = np.asarray(self._comp_mask(masks))
        masks = masks > 1e-5
        return masks

    def get_hog(self, x_o):
        use_bin = self.use_bin
        NO = self.NO
        BS = self.BS
        nc = self.nc
        x = (x_o + sharedX(1)) / (sharedX(2))
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4.0
        Gy = Gx.T
        f1_w = []
        for i in range(NO):
            t = np.pi / NO * i
            g = np.cos(t) * Gx + np.sin(t) * Gy
            gg = np.tile(g[np.newaxis, np.newaxis, :, :], [1, 1, 1, 1])
            f1_w.append(gg)
        f1_w = np.concatenate(f1_w, axis=0)
        G = np.concatenate([Gx[np.newaxis, np.newaxis, :, :], Gy[np.newaxis, np.newaxis, :, :]], axis=0)
        G_f = sharedX(floatX(G))

        a = np.cos(np.pi / NO)
        l1 = sharedX(floatX(1/(1-a)))
        l2 = sharedX(floatX(a/(1-a)))
        eps = sharedX(1e-3)
        if nc == 3:
            x_gray = T.mean(x, axis=1).dimshuffle(0, 'x', 1, 2)
        else:
            x_gray = x
        f1 = sharedX(floatX(f1_w))
        h0 = T.abs_(dnn_conv(x_gray, f1, subsample=(1, 1), border_mode=(1, 1)))
        g = dnn_conv(x_gray, G_f, subsample=(1, 1), border_mode=(1, 1))

        if use_bin:
            gx = g[:, [0], :, :]
            gy = g[:, [1], :, :]
            gg = T.sqrt(gx * gx + gy * gy + eps)
            hk = T.maximum(0, l1*h0-l2*gg)

            bf_w = np.zeros((NO, NO, 2*BS, 2*BS))
            b = 1 - np.abs((np.arange(1, 2 * BS + 1) - (2 * BS + 1.0) / 2.0) / BS)
            b = b[np.newaxis, :]
            bb = b.T.dot(b)
            for n in range(NO):
                bf_w[n,n] = bb

            bf = sharedX(floatX(bf_w))
            h_f = dnn_conv(hk, bf, subsample=(BS,BS), border_mode=(BS/2, BS/2))
            return h_f
        else:
            return g




