from __future__ import print_function
import sys
import argparse
import qdarkstyle
from PyQt4.QtGui import QApplication, QIcon
from PyQt4.QtCore import Qt
from ui import gui_design
from pydoc import locate
import constrained_opt

def parse_args():
    parser = argparse.ArgumentParser(description='iGAN: Interactive Visual Synthesis Powered by GAN')
    parser.add_argument('--model_name', dest='model_name', help='the model name', default='outdoor_64', type=str)
    parser.add_argument('--model_type', dest='model_type', help='the generative models and its deep learning framework', default='dcgan_theano', type=str)
    parser.add_argument('--framework', dest='framework', help='deep learning framework', default='theano')
    parser.add_argument('--win_size', dest='win_size', help='the size of the main window', type=int, default=384)
    parser.add_argument('--batch_size', dest='batch_size', help='the number of random initializations', type=int, default=64)
    parser.add_argument('--n_iters', dest='n_iters', help='the number of total optimization iterations', type=int, default=40)
    parser.add_argument('--top_k', dest='top_k', help='the number of the thumbnail results being displayed', type=int, default=16)
    parser.add_argument('--morph_steps', dest='morph_steps', help='the number of intermediate frames of morphing sequence', type=int, default=16)
    parser.add_argument('--model_file', dest='model_file', help='the file that stores the generative model', type=str, default=None)
    parser.add_argument('--d_weight', dest='d_weight', help='captures the visual realism based on GAN discriminator', type=float, default=0.0)
    parser.add_argument('--interp', dest='interp', help='the interpolation method (linear or slerp)', type=str, default='linear')
    parser.add_argument('--average', dest='average', help='averageExplorer mode',action="store_true", default=False)
    parser.add_argument('--shadow', dest='shadow', help='shadowDraw mode', action="store_true", default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if not args.model_file:  #if the model_file is not specified
        args.model_file = './models/%s.%s' % (args.model_name, args.model_type)

    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))

    args.win_size = int(args.win_size / 4.0) * 4  # make sure the width of the image can be divided by 4

    # initialize model and constrained optimization problem
    model_class = locate('model_def.%s' % args.model_type)
    model = model_class.Model(model_name=args.model_name, model_file=args.model_file)
    opt_class = locate('constrained_opt_%s' % args.framework)
    opt_solver = opt_class.OPT_Solver(model, batch_size=args.batch_size, d_weight=args.d_weight)
    img_size = opt_solver.get_image_size()
    opt_engine = constrained_opt.Constrained_OPT(opt_solver, batch_size=args.batch_size, n_iters=args.n_iters, topK=args.top_k,
                                                 morph_steps=args.morph_steps, interp=args.interp)

    # initialize application
    app = QApplication(sys.argv)
    window = gui_design.GUIDesign(opt_engine, win_size=args.win_size, img_size=img_size, topK=args.top_k,
                                  model_name=args.model_name, useAverage=args.average, shadow=args.shadow)
    app.setStyleSheet(qdarkstyle.load_stylesheet(pyside=False))  # comment this if you do not like dark stylesheet
    app.setWindowIcon(QIcon('pics/logo.png'))  # load logo
    window.setWindowTitle('Interactive GAN')
    window.setWindowFlags(window.windowFlags() & ~Qt.WindowMaximizeButtonHint)   # fix window siz
    window.show()
    app.exec_()