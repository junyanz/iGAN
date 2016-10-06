from __future__ import print_function
import sys
sys.path.append('..')
import os
from lib import utils
import argparse


# set parameters and arguments
parser = argparse.ArgumentParser('upgrade the old model to new model')
parser.add_argument('--old_model', dest='old_model', help='the path to the old model', default='../models/shoes_64.dcgan_theano', type=str)
parser.add_argument('--new_model', dest='new_model', help='the path to the new model', type=str, default=None)
args = parser.parse_args()

if not args.new_model:
    args.new_model = args.old_model.replace('.dcgan_theano', '_new.dcgan_theano')#new

for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

# load models
old_model = utils.PickleLoad(args.old_model)
# names = ['disc_params', 'gen_params', 'disc_batchnorm', 'gen_batchnorm', 'predict_params', 'predict_batchnorm']

model = {}
old_names = ['disc_params', 'gen_params', 'predict_params', 'postlearn_predict_params']
new_names = ['disc_params', 'gen_params', 'predict_params', 'predict_batchnorm']
for old_name, new_name in zip(old_names, new_names):
    model[new_name]=  old_model[old_name]
disc_batchnorm, gen_batchnorm= old_model['postlearn_params']
model['gen_batchnorm'] = gen_batchnorm
model['disc_batchnorm'] = disc_batchnorm

utils.PickleSave(args.new_model, model)
