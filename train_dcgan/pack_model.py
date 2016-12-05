from __future__ import print_function
import sys
sys.path.append('..')
import os
from lib import utils
import argparse

# set parameters and arguments
parser = argparse.ArgumentParser('compute batchnorm statistics for DCGAN model')
parser.add_argument('--model_name', dest='model_name', help='model name', default='shoes_64', type=str)
parser.add_argument('--cache_dir', dest='cache_dir', help='cache directory that stores models, samples and webpages', type=str, default=None)
parser.add_argument('--output_model', dest='output_model', help='output file that contains the compact model', type=str, default=None)
parser.add_argument('--ext', dest='ext', help='experiment name=model_name+ext', default='', type=str)
args = parser.parse_args()

expr_name = args.model_name + args.ext

if not args.cache_dir:
    args.cache_dir = './cache/%s/' % expr_name

model_dir = os.path.join(args.cache_dir, 'models')

if not args.output_model:
    args.output_model = os.path.join(args.cache_dir, '%s.dcgan_theano' % expr_name)


for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

# load models
model = {}
names = ['disc_params', 'gen_params', 'disc_batchnorm', 'gen_batchnorm', 'predict_params', 'predict_batchnorm']

for name in names:
    model_file = os.path.join(model_dir, name)
    if os.path.isfile(model_file):
        print('find model file %s' % model_file)
        model[name] = utils.PickleLoad(model_file)
    else:
        print('missing model file %s' % model_file)

utils.PickleSave(args.output_model, model)
