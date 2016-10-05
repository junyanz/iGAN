## Datasets

Download our hdf5 datasets (e.g. outdoor_64). The script will first download outdoor_64.zip, and then unzip it into outdoor_64.hdf5).
``` bash
bash ./datasets/scripts/download_hdf5_dataset.sh outdoor_64
```
* Outdoor natural images: [ourdoor_64.hdf5](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/datasets//outdoor_64.zip) (1.4G), [outdoor_128.hdf5](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/datasets//outdoor_128.zip) (5.5G), 150K landscape images from MIT [Places](http://places.csail.mit.edu/) dataset ([samples](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/samples/outdoor_64_real.png)).
* Outdoor church images: [church_64.hdf5](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/models/theano_dcgan/church_64.zip) (1.3G), [church_128.hdf5](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/models/theano_dcgan/church_128.zip) (4.6G), 126k church images from the [LSUN](http://lsun.cs.princeton.edu/2016/) challenge ([samples](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/samples/church_64_real.png)).
* Shoes images: [shoes_64.hdf5](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/models/theano_dcgan/shoes_64.zip) (260MB), [shoes_128.hdf5](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/models/theano_dcgan/shoes_128.zip) (922MB), 50K shoes images collected by [Yu and Grauman](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/) ([samples](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/samples/shoes_64_real.png)).
* Handbag images:  [handbag_64.hdf5](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/models/theano_dcgan/handbag_64.zip) (774MB), [handbag_128.hdf5](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/models/theano_dcgan/handbag_128.zip) (2.8G), 137K handbag images downloaded from Amazon ([samples](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/samples/handbag_64_real.png)).

### Sketch datasets
Download our sketch datasets (e.g. sketch_shoes_64)
* Shoes sketches: [sketch_shoes_64.hdf5](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/models/theano_dcgan/sketch_shoes_64.zip) (76MB), [sketch_shoes_128.hdf5](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/models/theano_dcgan/sketch_shoes_128.zip) (278MB), 50K shoes sketches collected by [Yu and Grauman](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/) and filtered by Photoshop sketch filter ([samples](https://people.eecs.berkeley.edu/~junyanz/projects/gvm/samples/sketch_shoes_64_real.png)).


## Training DCGAN model with your own dataset
Go to the directory `cd train_dcgan`:

Use the following script to create a hdf5 file from a directory of images. (See `python create_hdf5.py --help` for more details)
```bash
python create_hdf5.py --dataset_dir YOUR_OWN_FOLDER --width 64 --mode dir --channel 3 --hdf5_file images.hdf5
```

Install the following python libraries:
* [Scipy](http://www.scipy.org/install.html)
* [tqdm](https://github.com/noamraph/tqdm)  
```bash
sudo pip install tqdm
```
* [Fuel](https://fuel.readthedocs.io/en/latest/)
```bash
sudo pip install git+git://github.com/mila-udem/fuel.git
```

Train the model with a hdf5 file. (e.g. shoes_64.hdf5)
* Step 1: define the model parameters in `train_dcgan/train_dcgan_config.py` file
* Step 2: train a DCGAN  model
```bash
THEANO_FLAGS='device=gpu1, floatX=float32, nvcc.fastmath=True' python train_dcgan.py --class_name shoes_64
```
* Step 2: estimate the batchnorm parameters for DCGAN
```bash
THEANO_FLAGS='device=gpu1, floatX=float32, nvcc.fastmath=True' python batchnorm_dcgan.py --class_name shoes_64
```
* DCGAN_theano model on new datasets: we will provide a model training script soon (by Sep 25 2016). The script can train a model (e.g. cat_64.dcgan_theano) given a new photo collection. (e.g. cat_photos/)


## Reconstructing an Image:
Install [Lasagne](https://github.com/Lasagne/Lasagne)
```bash
sudo pip install https://github.com/Lasagne/Lasagne/archive/master.zip
```
Download AlexNet model (e.g. conv4):
```bash
bash models/scripts/download_alexnet.sh conv4
```


## Training the predictive network (`x->z`) on your own dataset


## Train a DCGAN model on your own dataset

## Train a generative model (e.g. VAE) based on Theano
The current design of our software follows: ui python class (e.g. `gui_draw.py`) => constrained optimization python class (`constrained_opt_theano.py`) => deep generative model python class (e.g. `dcgan_theano.py`). To incorporate your own generative model, you need to create a new python class (e.g. `vae_theano.py`) under `model_def` folder with the same interface of `dcgan_theano.py`, and specify `--model_type vae_theano` in the command line.

## Train a generative model based on Tensorflow
we are working on a tensorflow based optimization class (i.e. `constrained_opt_tensorflow.py`) now. Once the code is released, you can create your own tensorflow model class (e.g. `dcgan_tensorflow.py`) under `model_def` folder.
