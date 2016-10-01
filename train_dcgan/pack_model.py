import os
from lib import utils
import pickle

model_name = 'shirts_64'
print( 'pack model (%s)' % model_name )
model_fold = '../../models/%s' % model_name
output_model = '../../models/%s.dcgan_theano' % model_name

def load_postlearn(model_file):
    # st()
    with open(model_file, 'rb') as f:
        n_disc = pickle.load(f)
        bn_u = pickle.load(f)
        bn_s = pickle.load(f)
    gen_params = bn_u[:n_disc] + bn_s[:n_disc]
    disc_params = bn_u[n_disc:] + bn_s[n_disc:]
    return disc_params, gen_params
# load models
model = {}
names = ['disc_params', 'gen_params', 'postlearn_params', 'predict_params', 'postlearn_predict_params']
for name in names:
    if name == 'postlearn_params':
        model[name] = load_postlearn(os.path.join(model_fold, name))
    else:
        model[name] = utils.PickleLoad(os.path.join(model_fold, name))
utils.PickleSave(output_model, model)
