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
python train.py -t PATH_TO_IMAGE_SET -o saved_models/trained -b 128
python test.py  -t PATH_TO_IMAGE_SET -m saved_models/trained -n 128
```
where the image batch and sample sizes are specified via the `-b` and `-n` flags,
and the directory structure of the image set must look like the following:
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
