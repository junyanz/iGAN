import numpy as np

# import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from lib import utils
from lib.theano_utils import floatX, sharedX

NO = 8
BS = 8


def get_sobel(x):
    x1 = (x + sharedX(1)) / (sharedX(2))

    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4.0
    Gy = Gx.T
    Gx = Gx[np.newaxis, np.newaxis, :, :]
    Gy = Gy[np.newaxis, np.newaxis, :, :]
    G = np.concatenate([Gx, Gy], axis=0)
    x_gray = T.mean(x1, axis=1).dimshuffle(0, 'x', 1, 2)
    G_f = sharedX(floatX(G))

    g = dnn_conv(x_gray, G_f, subsample=(1, 1), border_mode=(1, 1))
    gx = g[:, [0], :, :]
    gy = g[:, [1], :, :]
    gg = gx * gx + gy * gy
    # gg = T.sqrt(gx * gx + gy * gy+1e-6)
    # gg = T.sqrt(gx * gx + gy * gy+1e-6)
    # blur the edge map

    # g_clip = T.clip(gg, 0, 0.5)*2
    # go = gg > 0.125
    # go = gx > 0.25
    return gg


def get_hog(x_o, use_bin=True, NO=8, BS=8):
    x = (x_o + sharedX(1)) / (sharedX(2))
    # gx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]) / 2.0
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

    # hk = gg * (h0 >  a* gg)
    # h_f = h0
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

#
# def vis_hog(hogs):
#     # im = ims[0]
#     hog = hogs[0]
#     n_chn = hog.shape[0]
#     # st()
#     n_grid = int(np.ceil(np.sqrt(n_chn)))
#
#     for n in range(n_chn):
#         hog_chn = hog[n]
#         plt.subplot(n_grid, n_grid, n + 1)
#         d = 180.0 / NO * n
#         plt.title('%3.3f degree' % d)
#         plt.imshow(hog_chn, cmap=plt.cm.Greys_r)
#         plt.colorbar()
#     plt.show()


m = T.tensor4()
# NO = self.NO
# BS = self.BS
bf_w = np.ones((1, 1, 2*BS, 2*BS))#1 - np.abs((np.arange(1,2*BS+1) - (2*BS+1.0) / 2.0) / BS)
bf = sharedX(floatX(bf_w))
m_b = dnn_conv(m, bf, subsample=(BS,BS), border_mode=(BS/2, BS/2))
compMask = theano.function(inputs=[m],outputs=m_b)

def comp_mask(masks):
    # masks = np.sign(masks)
    masks = np.asarray(compMask(masks))
    masks = masks > 1e-5
    # st()
    return masks

class HOGNet():
    # NO = 4
    # BS = 4

    def __init__(self):
        # print('COMPILING...')
        # t = time()
        # x = T.tensor4()
        # output = get_sobel(x)
        # self.__comp_edge = theano.function(inputs=[x], outputs=output)
        # print '%.2f seconds to compile __comp_edge function' % (time() - t)
        self.init_hog()
        #  dw = difn((outputf, inputf, fs, fs), 'dw%d' % (n+1))
        # self.init_hog()
        # self.init_comp_mask()

        # def init_comp_mask(self):
        # m = T.tensor4()
        # NO = self.NO
        # BS = self.BS
        # bf_w = np.ones((1, 1, 2 * BS, 2 * BS))  # 1 - np.abs((np.arange(1,2*BS+1) - (2*BS+1.0) / 2.0) / BS)
        # bf = sharedX(floatX(bf_w))
        # m_b = dnn_conv(m, bf, subsample=(BS, BS), border_mode=(BS / 2, BS / 2))
        # self.compMask = theano.function(inputs=[m], outputs=m_b)
        # b = b[np.newaxis,:]
        # bf_w = np.tile(b, (1, 1,1,1))
        # st()

    def comp_edge(self, ims):
        return self.__comp_edge(ims)
        #

    def init_hog(self):
        x = T.tensor4()
        h = get_hog(x)
        self.__comp_hog = theano.function(inputs=[x], outputs=h)

    def comp_hog(self, ims):
        utils.print_numpy(ims)
        print(ims.shape)
        # st()
        return np.asarray(self.__comp_hog(ims))

    def comp_mask(self, masks):
        # masks = np.sign(masks)
        masks = np.asarray(self.compMask(masks))
        masks = floatX(masks > 1e-5)
        # print(masks.shape)



    # def vis_edge(self, ims, edgs):
    #     im = ims[0, 0]
    #     edg = edgs[0, 0]
    #     # st()
    #     plt.subplot(2, 1, 1)
    #     plt.imshow(im, cmap=plt.cm.Greys_r)
    #     plt.colorbar()
    #     plt.subplot(2, 1, 2)
    #     plt.imshow(edg, cmap=plt.cm.Greys_r)
    #     plt.colorbar()
    #     plt.show()
    #
    # def vis_mask(self, mask):
    #     mask_im = mask[0, 0, :, :]
    #     plt.imshow(mask_im)
    #     plt.show()
    #     plt.waitforbuttonpress()

