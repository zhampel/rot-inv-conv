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
    num_classes, height, width, channels = dir_struct.get_img_data()

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
    num_classes, height, width, channels = dir_struct.get_img_data()

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


def test_fixed_rot_img_generator(dir_struct=None, batch_size=32, rotation_angle=0.):
    """
    Function to define test generator
    for testing CNN model.

    Parameters
    ----------
    dir_struct        : DataDirStruct dictionary
                        Data directory definitions
    batch_size        : int
                        batch size for training
    rotation_angle    : float
                        fixed angle for rotating images

    Returns
    -------
    test_generator    : generator
                        generator object for test data
    """

    # Get image data from .dat file
    num_classes, height, width, channels = dir_struct.get_img_data()

    target_size = (height, width)
    
    if not any(target_size):
        raise ValueError('Invalid image dimensions {}. '
                         'All values must be >0.'.format(target_size))

    # Define preprocessing function with fixed angle image rotation
    img_shape = (height, width, channels)
    my_img_processor = MyPreProcessor(img_shape=img_shape, rescale=1./255, fixed_rot_angle_deg=rotation_angle)
    preprocess_img = my_img_processor.preprocess_img

    # Instantiate Data Generator with custom preprocessing function
    datagen = ImageDataGenerator(preprocessing_function=preprocess_img)

    # For testing
    test_generator = datagen.flow_from_directory(dir_struct.test_dir,
                                                 target_size=target_size,
                                                 batch_size=batch_size)

    return test_generator

# Custom preprocessing class to feed fixed rotation angle
# to ImageDataGenerator()'s apply_transform method
class MyPreProcessor(object):
    def __init__(self, img_shape=None, rescale=1./255, fixed_rot_angle_deg=0.):
        self.rescale     = rescale
        self.rot_angle   = fixed_rot_angle_deg
        self.img_shape   = img_shape
        self.datagen     = ImageDataGenerator()
        self.transform_parameters = {'theta' : fixed_rot_angle_deg}

        # Choose rotation function based on # image channels
        if (self.img_shape[2] > 1):
            self.rot_image = self.three_channel_rot
        else:
            self.rot_image = self.one_channel_rot

        print('\n\nLoading custom preprocessing function on images of size {}'
              ' with a fixed rotation of {} deg'.format(self.img_shape, self.rot_angle))

    def one_channel_rot(self, img):
        # Padding: Keras requires 3D tensor for 2D image
        padded_img = np.zeros((self.img_shape[0], self.img_shape[1], 1))
        padded_img[:,:,0] = img.copy()

        # Rotate Image
        rot_img = self.datagen.apply_transform(padded_img, self.transform_parameters)
        return rot_img[:,:,0]

    def three_channel_rot(self, img):
        # Padding: Keras requires 3D tensor for 2D image
        padded_img = np.zeros((self.img_shape[0], self.img_shape[1], 1))
        proc_img = np.zeros(self.img_shape)

        # Loop through channels
        for i in range(self.img_shape[2]):
            padded_img[:,:,0] = img[:,:,i].copy()
            # Rotate image channel
            rot_img = self.datagen.apply_transform(padded_img, self.transform_parameters)
            # Save for output
            proc_img[:,:,i] = rot_img[:,:,0]

        return proc_img



    def preprocess_img(self, img):
        """
        Image preprocessing function
        to include rescaling and fixed rotations.
        Can only have one input, i.e. the image, per
        https://keras.io/preprocessing/image/#imagedatagenerator-class

        Parameters
        ----------
        img        : keras image
                     image object

        Returns
        -------
        proc_img   : image array
                     preprocessed image
        """

        # Scale image
        scale_img = img.astype(np.float32) * self.rescale

        return self.rot_image(scale_img)
