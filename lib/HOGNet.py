import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
from lib.theano_utils import floatX, sharedX

NO = 8
BS = 8

def get_hog(x_o, use_bin=True, NO=8, BS=8):
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

    # st()
    a = np.cos(np.pi / NO)
    l1 = sharedX(floatX(1/(1-a)))
    l2 = sharedX(floatX(a/(1-a)))
    eps = sharedX(1e-3)
    # x = T.tensor4()
    x_gray = T.mean(x, axis=1).dimshuffle(0, 'x', 1, 2)
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


m = T.tensor4()
bf_w = np.ones((1, 1, 2*BS, 2*BS))
bf = sharedX(floatX(bf_w))
m_b = dnn_conv(m, bf, subsample=(BS,BS), border_mode=(BS/2, BS/2))
compMask = theano.function(inputs=[m],outputs=m_b)

def comp_mask(masks):
    masks = np.asarray(compMask(masks))
    masks = masks > 1e-5
    return masks
