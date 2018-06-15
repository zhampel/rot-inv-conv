# rot-inv-conv
Towards a rotationally invariant convolutional layer.

This package provides a TensorFlow Convolution2D layer that introduces invariance to image rotation.


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




## License

[MIT License](LICENSE)

Copyright (c) 2018 Zigfried Hampel-Arias
