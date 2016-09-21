import os
import utils
import cPickle
from theano_utils import floatX, sharedX
from pdb import set_trace as st

model_name = 'shirts_64'
print 'pack model (%s)' % model_name
model_fold = '../../models/%s' % model_name
output_model = '../../models/%s.dcgan_theano' % model_name

def load_postlearn(model_file):
    # st()
    with open(model_file, 'rb') as f:
        n_disc = cPickle.load(f)
        bn_u = cPickle.load(f)
        bn_s = cPickle.load(f)
        # bn_u = [sharedX(d) for d in cPickle.load(f)]
        # bn_s = [sharedX(d) for d in cPickle.load(f)]
    gen_params = bn_u[:n_disc] + bn_s[:n_disc]
    disc_params = bn_u[n_disc:] + bn_s[n_disc:]
    return disc_params, gen_params
# load models
model = {}
# names = ['postlearn_params']
names = ['disc_params', 'gen_params', 'postlearn_params', 'predict_params', 'postlearn_predict_params']
for name in names:
    if name == 'postlearn_params':
        model[name] = load_postlearn(os.path.join(model_fold, name))
        # st()
    else:
        model[name] = utils.PickleLoad(os.path.join(model_fold, name))
# st()
utils.PickleSave(output_model, model)
