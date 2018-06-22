# rot-inv-conv
Towards a rotationally invariant convolutional layer.

This package provides a TensorFlow Convolution2D layer that introduces invariance to image rotation.

To check-out the repo:
```
git clone https://github.com/zhampel/rot-inv-conv.git
```

## Main Idea
A purely translational convolution slides a kernel operation over an image, like so:

![cnn-sliding-kernel](docs/images/cnn-sliding.gif)

By including a rotation of the same kernel, we can search for the maximal activation
given the kernel orientation.
The operation using 90 degree turns can be visualized as follows:

![cnn-4rot-kernel](docs/images/cnn-4rot.gif)

The operation using 45 degree turns looks like this:

![cnn-8rot-kernel](docs/images/cnn-8rot.gif)


## Requirements
The package as well as the necessary requirements can be installed via
```
python setup.py install
```

## Run a training and test
To train and test a model on a set of images, 
```
python scripts/train.py -c config.yml
python scripts/test.py  -c config.yml -m model1
```
where the YAML config file is specified via the `-c` flag, and the model to test is given by `-m`.
The directory structure of the image data set must look like the following:
```
dataset/
│   img_data.dat
└───training/
│   0/
│   1/
│  ...
│   n_classes/
└───testing/
│   0/
│   1/
│  ...
│   n_classes/
```


The img_data.dat file contains in CSV the following values: number of classes, height, width, channels.
For example, for the CIFAR-10 data set, the contents of img_data.dat is `10,32,32,3`.

To start with a standard data set, one can run the following:
```
cd scripts/sample_data
python save_cifar10.py
```
which will download the CIFAR-10 data set, convert the images to pngs, and save them
with the required directory structure above.


One can also download the MNIST data set
from [here](http://yann.lecun.com/exdb/mnist/), then convert them to pngs via
```
cd scripts/sample_data
python convert_mnist_to_png.py PATH_TO_MNIST_GZIPS DESIRED_PATH_MNIST_PNGS
```


## License

[MIT License](LICENSE)

Copyright (c) 2018 Zigfried Hampel-Arias
