from __future__ import print_function
import argparse
from pydoc import locate
import constrained_opt
import cv2
import numpy as np
from pdb import set_trace as st

def parse_args():
    parser = argparse.ArgumentParser(description='iGAN: Interactive Visual Synthesis Powered by GAN')
    parser.add_argument('--model_name', dest='model_name', help='the model name', default='outdoor_64', type=str)
    parser.add_argument('--model_type', dest='model_type', help='the generative models and its deep learning framework', default='dcgan_theano', type=str)
    parser.add_argument('--framework', dest='framework', help='deep learning framework', default='theano')
    parser.add_argument('--input_color', dest='input_color', help='input color image', default='./pics/input_color.png')
    parser.add_argument('--input_color_mask', dest='input_color_mask', help='input color mask', default='./pics/input_color_mask.png')
    parser.add_argument('--input_edge', dest='input_edge', help='input edge image', default='./pics/input_edge.png')
    parser.add_argument('--output_result', dest='output_result', help='output_result', default='./pics/script_result.png')
    parser.add_argument('--batch_size', dest='batch_size', help='the number of random initializations', type=int, default=64)
    parser.add_argument('--n_iters', dest='n_iters', help='the number of total optimization iterations', type=int, default=100)
    parser.add_argument('--top_k', dest='top_k', help='the number of the thumbnail results being displayed', type=int, default=16)
    parser.add_argument('--model_file', dest='model_file', help='the file that stores the generative model', type=str, default=None)
    parser.add_argument('--d_weight', dest='d_weight', help='captures the visual realism based on GAN discriminator', type=float, default=0.0)
    args = parser.parse_args()
    return args

def preprocess_image(img_path, npx):
    im = cv2.imread(img_path, 1)
    if im.shape[0] != npx or im.shape[1] != npx:
        out = cv2.resize(im, (npx, npx))
    else:
        out = np.copy(im)

    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out
if __name__ == '__main__':
    args = parse_args()
    if not args.model_file:  #if the model_file is not specified
        args.model_file = './models/%s.%s' % (args.model_name, args.model_type)

    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))

    # initialize model and constrained optimization problem
    model_class = locate('model_def.%s' % args.model_type)
    model = model_class.Model(model_name=args.model_name, model_file=args.model_file)
    opt_class = locate('constrained_opt_%s' % args.framework)
    opt_solver = opt_class.OPT_Solver(model, batch_size=args.batch_size, d_weight=args.d_weight)
    img_size = opt_solver.get_image_size()
    opt_engine = constrained_opt.Constrained_OPT(opt_solver, batch_size=args.batch_size, n_iters=args.n_iters, topK=args.top_k)
    # load user inputs
    npx = model.npx
    im_color = preprocess_image(args.input_color, npx)
    im_color_mask = preprocess_image(args.input_color_mask, npx)
    im_edge = preprocess_image(args.input_edge, npx)
    # run the optimization
    opt_engine.init_z()
    constraints = [im_color, im_color_mask[... ,[0]], im_edge, im_edge[...,[0]]]
    for n in range(args.n_iters):
        opt_engine.update_invert(constraints=constraints)
    results = opt_engine.get_current_results()
    final_result= np.concatenate(results, 1)
    # combine input and output
    final_vis = np. hstack([im_color, im_color_mask, im_edge, final_result])
    final_vis = cv2.cvtColor(final_vis, cv2.COLOR_RGB2BGR)
    final_vis = cv2.resize(final_vis, (0, 0), fx=2.0, fy=2.0)
    # save
    cv2.imwrite(args.output_result, final_vis)
