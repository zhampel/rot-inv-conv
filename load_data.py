from __future__ import print_function
import sys
import os.path
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


def train_img_generator(dir_struct=None, batch_size=32, rotation_range=0., val_split=0.2):
    """
    Function to define training and validation generators
    for training CNN model.

    Parameters
    ----------
    dir_struct        : DataDirStruct dictionary
                        Data directory definitions
    batch_size        : int
                        batch size for training
    rotation_range    : float
                        range for rotating images
    val_split         : float
                        fraction of data set used
                        for validation

    Returns
    -------
    train_generator   : generator
                        generator object for training data

    val_generator     : generator
                        generator object for validation data
    """

    # Get image data from .dat file
    num_classes, height, width = dir_struct.get_img_data()

    target_size = (height, width)
    
    if not any(target_size):
        raise ValueError('Invalid image dimensions {}. '
                         'All values must be >0.'.format(target_size))

    # Instantiate Data Generator
    # only augmentation is to rescale
    datagen = ImageDataGenerator(rescale=1./255, \
                                 rotation_range=rotation_range, \
                                 validation_split=val_split)

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


def test_img_generator(dir_struct=None, batch_size=32, rotation_range=0.):
    """
    Function to define test generator
    for testing CNN model.

    Parameters
    ----------
    dir_struct        : DataDirStruct dictionary
                        Data directory definitions
    batch_size        : int
                        batch size for training
    rotation_range    : float
                        range for rotating images

    Returns
    -------
    test_generator    : generator
                        generator object for test data
    """

    # Get image data from .dat file
    num_classes, height, width = dir_struct.get_img_data()

    target_size = (height, width)
    
    if not any(target_size):
        raise ValueError('Invalid image dimensions {}. '
                         'All values must be >0.'.format(target_size))

    # Instantiate Data Generator
    # only augmentation is to rescale
    datagen = ImageDataGenerator(rescale=1./255, rotation_range=rotation_range)

    # For testing
    test_generator = datagen.flow_from_directory(dir_struct.test_dir,
                                                 target_size=target_size,
                                                 batch_size=batch_size)

    return test_generator
