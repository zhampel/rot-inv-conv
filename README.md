# rot-inv-conv
Towards a rotationally invariant convolutional layer.

This package provides a TensorFlow Convolution2D layer that introduces invariance to image rotation.

To check-out the repo:
```
git clone https://github.com/zhampel/rot-inv-conv.git
```

## Main Idea
A purely translational convolution slides a kernel operation over an image, like so:

![cnn-sliding-kernel](images/cnn-sliding.gif =250x)

By including a rotation of the same kernel, we can search for the maximal activation
given the kernel orientation.
The operation using 90 degree turns can be visualized as follows:

![cnn-4rot-kernel](images/cnn-4rot.gif =250x)

The operation using 45 degree turns looks like this:

![cnn-4rot-kernel](images/cnn-8rot.gif)


## Requirements
The required packages to run the training and testing scripts can be installed via 
```
pip install -r requirements.txt
```

## Run a training and test
To train and test a model on a set of images, 
```
python train.py -c config.yml
python test.py  -c config.yml -m model1 -n 128
```
where the YAML config file is specified via the `-c` flag, and the model to test over
and the number of samples to test with are given by `-m` and `-n`, respectively.
The directory structure of the image data set must look like the following:
- PATH_TO_IMAGE_SET
  - training
    - 0
    - 1
    ...
    - N_CLASSES
  - testing
    - 0
    - 1
    ...
    - N_CLASSES


To start with a standard data set, one can run the following:
```
cd sample_data
python save_cifar10.py
```
which will download the CIFAR-10 data set, convert the images to pngs, and save them
with the required directory structure above.


One can also download the MNIST data set
from [here](http://yann.lecun.com/exdb/mnist/), then convert them to pngs via
```
cd sample_data
python convert_mnist_to_png.py PATH_TO_MNIST_GZIPS DESIRED_PATH_MNIST_PNGS
```


## License

[MIT License](LICENSE)

Copyright (c) 2018 Zigfried Hampel-Arias
