import theano
import theano.tensor as T


def CategoricalCrossEntropy(y_pred, y_true):
    return T.nnet.categorical_crossentropy(y_pred, y_true).mean()


def BinaryCrossEntropy(y_pred, y_true):
    return T.nnet.binary_crossentropy(y_pred, y_true).mean()


def L2Loss(y_pred, y_true):
    return T.sqr(y_pred - y_true).mean()


def L1Loss(y_pred, y_true):
    return T.abs_(y_pred - y_true).mean()


def MaskedL1Loss(y_pred, y_true, m):
    return (T.abs_(y_pred - y_true) * m).mean() / m.mean()

def MaskedL2Loss(y_pred, y_true, m):
    return (T.sqr(y_pred - y_true) * m).mean() / m.mean()


def TruncatedL1(y_pred, y_true, tr):
    return T.maximum(T.abs_(y_pred - y_true), tr).mean()


def SquaredHinge(y_pred, y_true):
    return T.sqr(T.maximum(1. - y_true * y_pred, 0.)).mean()


def Hinge(y_pred, y_true):
    return T.maximum(1. - y_true * y_pred, 0.).mean()


# cce = CCE = CategoricalCrossEntropy
bce = BinaryCrossEntropy
#mse = MSE = MeanSquaredError
#mae = MAE = MeanAbsoluteError
