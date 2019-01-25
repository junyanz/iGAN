## Datasets

Download our hdf5 datasets (e.g. outdoor_64). The script will first download outdoor_64.zip, and then unzip it into outdoor_64.hdf5).
``` bash
bash ./datasets/scripts/download_hdf5_dataset.sh outdoor_64
```
* Outdoor natural images: [ourdoor_64.hdf5](http://efrosgans.eecs.berkeley.edu/iGAN/datasets/outdoor_64.zip) (1.4G), [outdoor_128.hdf5](http://efrosgans.eecs.berkeley.edu/iGAN/datasets/outdoor_128.zip) (5.5G), 150K landscape images from MIT [Places](http://places.csail.mit.edu/) dataset ([samples](http://efrosgans.eecs.berkeley.edu/iGAN/samples/outdoor_64_real.png)).
* Outdoor church images: [church_64.hdf5](http://efrosgans.eecs.berkeley.edu/iGAN/datasets/church_64.zip) (1.3G), [church_128.hdf5](http://efrosgans.eecs.berkeley.edu/iGAN/datasets/church_128.zip) (4.6G), 126k church images from the [LSUN](http://lsun.cs.princeton.edu/2016/) challenge ([samples](http://efrosgans.eecs.berkeley.edu/iGAN/samples/church_64_real.png)).
* Shoes images: [shoes_64.hdf5](http://efrosgans.eecs.berkeley.edu/iGAN/datasets/shoes_64.zip) (260MB), [shoes_128.hdf5](http://efrosgans.eecs.berkeley.edu/iGAN/datasets/shoes_128.zip) (922MB), 50K shoes images collected by [Yu and Grauman](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/) ([samples](http://efrosgans.eecs.berkeley.edu/iGAN/samples/shoes_64_real.png)).
* Handbag images:  [handbag_64.hdf5](http://efrosgans.eecs.berkeley.edu/iGAN/datasets/handbag_64.zip) (774MB), [handbag_128.hdf5](http://efrosgans.eecs.berkeley.edu/iGAN/datasets/handbag_128.zip) (2.8G), 137K handbag images downloaded from Amazon ([samples](http://efrosgans.eecs.berkeley.edu/iGAN/samples/handbag_64_real.png)).

### Sketch datasets
Download the sketch datasets (e.g. sketch_shoes_64)
* Shoes sketches: [sketch_shoes_64.hdf5](http://efrosgans.eecs.berkeley.edu/iGAN/datasets/sketch_shoes_64.zip) (76MB), [sketch_shoes_128.hdf5](http://efrosgans.eecs.berkeley.edu/iGAN/datasets/sketch_shoes_128.zip) (278MB), [sketch_shoes_64.hdf5](http://efrosgans.eecs.berkeley.edu/iGAN/datasets/hed_shoes_64.zip) (69MB), [sketch_shoes_128.hdf5](http://efrosgans.eecs.berkeley.edu/iGAN/datasets/hed_shoes_128.zip) (244MB), 50K shoes sketches collected by [Yu and Grauman](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/) and filtered by Photoshop sketch filter or [HED](https://github.com/s9xie/hed) edge detection.  ([Photoshop sketch samples](http://efrosgans.eecs.berkeley.edu/iGAN/samples/sketch_shoes_64_real.png), [HED samples](http://efrosgans.eecs.berkeley.edu/iGAN/samples/hed_shoes_64_real.png)).


## Train a DCGAN model
* Install the following python libraries:
  * [Scipy](http://www.scipy.org/install.html)
  * [tqdm](https://github.com/noamraph/tqdm)  
  ```bash
  sudo pip install tqdm
  ```


  * Install [h5py](http://docs.h5py.org/en/latest/build.html) and [Fuel](https://fuel.readthedocs.io/en/latest/)
  ```bash
  sudo pip install h5py
  sudo pip install git+git://github.com/mila-udem/fuel.git
  ```

* Train the model with a hdf5 file. (e.g. shoes_64.hdf5)
  * Go the training code directory:
  ```bash
  cd train_dcgan
  ```
  * Define the model parameters in `train_dcgan_config.py` file.
  * Train a DCGAN  model:
  ```bash
  THEANO_FLAGS='device=gpu0, floatX=float32, nvcc.fastmath=True' python train_dcgan.py --model_name shoes_64
    ```
  By default, the training code will create a directory `./cache/`, and store all the generated samples, webpage, and model checkpoints in the directory.

  * Estimate the batchnorm parameters for DCGAN:
  ```bash
  THEANO_FLAGS='device=gpu0, floatX=float32, nvcc.fastmath=True' python batchnorm_dcgan.py --model_name shoes_64
  ```
  * (Optionally) Train an inverting model to predict `z` from `x`:
  ```bash
  THEANO_FLAGS='device=gpu0, floatX=float32, nvcc.fastmath=True' python train_predict_z.py --model_name shoes_64
  ```
  * (Optionally) Estaimate the batchnorm parameters for the inverting model:
  ```bash
  THEANO_FLAGS='device=gpu0, floatX=float32, nvcc.fastmath=True' python batchnorm_predict_z.py --model_name shoes_64
  ```
  * Pack the model:
  ```bash
  python pack_model.py --model_name shoes_64 --output_model shoes_64.dcgan_theano
  ```

* Train a model on your own dataset
  * Run the script to create a hdf5 file from an image collection (see `python create_hdf5.py --help` for more details):
  ```bash
  python create_hdf5.py --dataset_dir YOUR_OWN_FOLDER --width 64 --mode dir --channel 3 --hdf5_file images.hdf5
  ```

## Train a generative model (e.g. VAE) based on Theano

The current design of our software follows: ui python class (e.g. `gui_draw.py`) => python wrapper for constrained optimization  (`constrained_opt.py`) => Theano implementation of constrained optimization (`constrained_opt_theano.py`) => deep generative model implemented in Theano (e.g. `dcgan_theano.py`). To incorporate your own generative model, you need to create a new python class (e.g. `vae_theano.py`) under `model_def` folder with the same interface of `dcgan_theano.py`, and specify `--model_type vae_theano` in the command line.
## Train a generative model based on Tensorflow
we are working on a tensorflow based optimization class (i.e. `constrained_opt_tensorflow.py`) now. Once the code is released, you can create your own tensorflow model class (e.g. `dcgan_tensorflow.py`) under `model_def` folder.
