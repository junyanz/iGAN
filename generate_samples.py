from __future__ import print_function
import argparse
from pydoc import locate
from lib import utils
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='generated random samples (dcgan_theano)')
    parser.add_argument('--model_name', dest='model_name', help='the model name', default='outdoor_64', type=str)
    parser.add_argument('--model_type', dest='model_type', help='the generative models and its deep learning framework', default='dcgan_theano', type=str)
    parser.add_argument('--framework', dest='framework', help='deep learning framework', default='theano')
    parser.add_argument('--model_file', dest='model_file', help='the file that stores the generative model', type=str, default=None)
    parser.add_argument('--output_image', dest='output_image', help='the name of output image', type=str, default=None)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if not args.model_file:  #if model directory is not specified
        args.model_file = './models/%s.%s' % (args.model_name, args.model_type)

    if not args.output_image:
        args.output_image = '%s_%s_samples.png' % (args.model_name, args.model_type)

    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))

    # initialize model and constrained optimization problem
    model_class = locate('model_def.%s' % args.model_type)
    model = model_class.Model(model_name=args.model_name, model_file=args.model_file)
    # generate samples
    samples = model.gen_samples(z0=None, n=196, batch_size=49, use_transform=True)
    # generate grid visualization
    im_vis = utils.grid_vis(samples, 14, 14)
    # write to the disk
    im_vis = cv2.cvtColor(im_vis, cv2.COLOR_BGR2RGB)
    cv2.imwrite(args.output_image, im_vis)
    print('samples_shape', samples.shape)
    print('save image to %s' % args.output_image)