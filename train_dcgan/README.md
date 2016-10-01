## Our dataset

Download our dataset (hdf5) (e.g. outdoor_64.hdf5).
``` bash
bash ./datasets/scripts/download_hdf5_dataset.sh outdoor_64.hdf5
```
* [ourdoor_64.hdf5](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/datasets//outdoor_64.hdf5) (64x64, 1.8G): trained on 150K landscape images from MIT [Places](http://places.csail.mit.edu/) dataset ([samples](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/samples/outdoor_64_real.png)).
* [shoes_64.hdf5](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/models/theano_dcgan/shoes_64.hdf5) (64x64): trained on 50K shoes images collected by [Yu and Grauman](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/) ([samples](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/samples/shoes_64_real.png)).
* [church_64.hdf5](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/models/theano_dcgan/church_64.hdf5) (64x64): 126k church images from the [LSUN](http://lsun.cs.princeton.edu/2016/) challenge ([samples](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/samples/church_64_real.png)).
* [handbag_64.hdf5](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/models/theano_dcgan/handbag_64.hdf5) (64x64): 137K handbag images downloaded from Amazon ([samples](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/samples/handbag_64_real.png)).



## Training on your own dataset
Use the following script to create a hdf5 file from a directory of images.
```bash
```
Train the model on hdf5 file.
```bash
```

* DCGAN_theano model on new datasets: we will provide a model training script soon (by Sep 25 2016). The script can train a model (e.g. cat_64.dcgan_theano) given a new photo collection. (e.g. cat_photos/)



## Train a DCGAN model on your own dataset

## Train a generative model (e.g. VAE) based on Theano
The current design of our software follows: ui python class (e.g. `gui_draw.py`) => constrained optimization python class (`constrained_opt_theano.py`) => deep generative model python class (e.g. `dcgan_theano.py`). To incorporate your own generative model, you need to create a new python class (e.g. `vae_theano.py`) under `model_def` folder with the same interface of `dcgan_theano.py`, and specify `--model_type vae_theano` in the command line.

## Train a generative model based on Tensorflow
we are working on a tensorflow based optimization class (i.e. `constrained_opt_tensorflow.py`) now. Once the code is released, you can create your own tensorflow model class (e.g. `dcgan_tensorflow.py`) under `model_def` folder.
