from __future__ import print_function
import sys
import os.path
import numpy as np
import cv2
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

def load_single_image(filepath=''):
    try:
        open(filepath, 'r')
    except IOError:
        print('File %s does not exist!'%filepath)
        return

    # Extract image file
    img = cv2.imread(filepath, 0)
    
    # Normalize
    img = img.astype('float32')
    img /= 255
    shape = img.shape

    return img, shape


## Rotate image by angle given in degrees
def rotate_image(img, rot_ang_deg=0.):

    # Using Keras data augmenting functions
    datagen = ImageDataGenerator()
    transform_parameters = {'theta' : rot_ang_deg}

    # Padding: Keras requires 3D tensor for 2D image
    padded_img = np.zeros((img.shape[0], img.shape[1], 1))
    padded_img[:,:,0] = img.copy()

    # Rotate Image
    rot_img = datagen.apply_transform(padded_img, transform_parameters)

    return rot_img[:,:,0]

def train_img_generator(dir_struct=None, batch_size=32, val_split=0.2):

    # Get image data from .dat file
    num_classes, height, width = dir_struct.get_img_data()

    target_size = (height, width)
    
    if not any(target_size):
        raise ValueError('Invalid image dimensions {}. '
                         'All values must be >0.'.format(target_size))

    # Instantiate Data Generator
    # only augmentation is to rescale
    datagen = ImageDataGenerator(rescale=1./255, validation_split=val_split)

    # For training
    train_generator = datagen.flow_from_directory(dir_struct.train_dir,
                                                  target_size=target_size,
                                                  batch_size=batch_size,
                                                  subset='training')

    # For validation
    val_generator = datagen.flow_from_directory(dir_struct.train_dir,
                                                target_size=target_size,
                                                batch_size=batch_size,
                                                subset='validation')

    if num_classes != train_generator.num_classes:
        raise ValueError('Expected number of classes "{}" '
                         'not equal to that found {} in file "{}".'.format(train_generator.num_classes, \
                                                                           dir_struct.data_file, \
                                                                           num_classes))

    return train_generator, val_generator


def test_img_generator(dir_struct=None, batch_size=32):

    # Get image data from .dat file
    num_classes, height, width = dir_struct.get_img_data()

    target_size = (height, width)
    
    if not any(target_size):
        raise ValueError('Invalid image dimensions {}. '
                         'All values must be >0.'.format(target_size))

    # Instantiate Data Generator
    # only augmentation is to rescale
    datagen = ImageDataGenerator(rescale=1./255)

    # For testing
    test_generator = datagen.flow_from_directory(dir_struct.test_dir,
                                                 target_size=target_size,
                                                 batch_size=batch_size)

    return test_generator
