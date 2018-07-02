# Convolutions with Rotational Operation
Towards a rotationally invariant convolutional layer.

This package provides a TensorFlow Convolution2D layer that introduces invariance to image rotation.

To check-out the repo:
```
git clone https://github.com/zhampel/rot-inv-conv.git
```

## Main Idea
Consider a simple 3x3 convolution kernel that has an L-shape,
shown in the animations below.
The standard 2D convolution slides this kernel operation over an image.
This purely translational convolution is shown in the left panel.

By including a rotation of the same kernel, we can search for the maximal activation
given the kernel orientations.
The operation using 90 degree turns is visualized in the center panel.
The operation with 45 degree turns is shown in the right panel.

<p float="left">
  <img src="docs/images/cnn-sliding.gif" width="200" />
  <img src="docs/images/cnn-4rot.gif" width="200" />
  <img src="docs/images/cnn-8rot.gif" width="200" />
</p>

### Presentation Slides
Further details regarding the motivation, methods and results of implementing
this idea can be found in my presentation 
<a href="http://bit.ly/zhampel-rot-inv-conv-slides" target="_blank">here</a>.


## Requirements
The package as well as the necessary requirements can be installed by running `make` or via
```
python setup.py install
```

## Run a training and test
To train and test a model on a set of images, 
```
python scripts/train.py -c config.yml
python scripts/test_batch.py  -c config.yml -m model1 -n 1000
```
where the YAML config file is specified via the `-c` flag, the model to test is given by `-m`,
and the `-n` supplies the number of images to test.
The directory structure of the image data set must look like the following:
```
dataset/
│
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

One can also run inference on images given a trained model (model1) and a set of (0, 90, 180) deg rotations via the following:
```
python scripts/predict_images.py -c config.yml -f 0 90 180 -i IMAGE.png -m model1
```


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
