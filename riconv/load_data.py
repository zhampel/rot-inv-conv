from __future__ import print_function
import sys
import os.path
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

# Color mode dictionary for specifying 
# color_mode in data generators
color_mode_dict = {1 : 'grayscale',
                   3 : 'rgb'}


def train_img_generator(dir_struct=None, config_struct=None):
    """
    Function to define training and validation generators
    for training CNN model.

    Parameters
    ----------
    dir_struct        : DataDirStruct dictionary
                        Data directory definitions
    config_struct     : ModelConfigurator dictionary
                        Runtime configuration

    Returns
    -------
    train_generator   : generator
                        generator object for training data

    val_generator     : generator
                        generator object for validation data
    """

    # Get image data from configurator
    num_classes    = config_struct.classes
    height         = config_struct.height
    width          = config_struct.width
    channels       = config_struct.channels
    batch_size     = config_struct.batch_size
    val_split      = config_struct.val_split
    rotation_range = config_struct.rotation_range

    color_mode = color_mode_dict[channels]

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
                                                  color_mode=color_mode,
                                                  batch_size=batch_size,
                                                  subset='training')

    # For validation
    val_generator = datagen.flow_from_directory(dir_struct.train_dir,
                                                target_size=target_size,
                                                color_mode=color_mode,
                                                batch_size=batch_size,
                                                subset='validation')

    if num_classes != train_generator.num_classes:
        raise ValueError('Expected number of classes "{}" '
                         'not equal to that found {} in configuration file "{}".'.format(train_generator.num_classes,
                                                                                         config_struct.filepath,
                                                                                         num_classes))

    return train_generator, val_generator


def test_img_generator(dir_struct=None, config_struct=None, fixed_rotation=False, rotation_angle=0., save_to_dir=None, save_prefix=''):
    """
    Function to define test generator
    for testing CNN model.

    Parameters
    ----------
    dir_struct        : DataDirStruct dictionary
                        Data directory definitions
    config_struct     : ModelConfigurator dictionary
                        Runtime configuration
    fixed_rotation    : bool
                        flag for fixed or random
                        angle range of image rotations
    rotation_angle    : float
                        range for rotating images

    Returns
    -------
    test_generator    : generator
                        generator object for test data
    """

    # Get image data from configurator
    num_classes    = config_struct.classes
    height         = config_struct.height
    width          = config_struct.width
    channels       = config_struct.channels
    batch_size     = config_struct.batch_size
    
    color_mode = color_mode_dict[channels]

    target_size = (height, width)
    
    if not any(target_size):
        raise ValueError('Invalid image dimensions {}. '
                         'All values must be >0.'.format(target_size))

    if fixed_rotation:
        print("Fixed rotation angle: {} deg".format(rotation_angle))
        # Define preprocessing function with fixed angle image rotation
        img_shape = (height, width, channels)
        my_img_processor = MyPreProcessor(img_shape=img_shape,
                                          rescale=1./255, 
                                          fixed_rot_angle_deg=rotation_angle)
        preprocess_img = my_img_processor.preprocess_img

        # Instantiate Data Generator with custom preprocessing function
        datagen = ImageDataGenerator(preprocessing_function=preprocess_img)


    else:
        print("Random rotation angle range: (-{}, {}) deg".format(rotation_angle, rotation_angle))
        # Instantiate Data Generator: rescale and random rotation
        datagen = ImageDataGenerator(rescale=1./255, rotation_range=rotation_angle)

    # For testing
    test_generator = datagen.flow_from_directory(dir_struct.test_dir,
                                                 target_size=target_size,
                                                 color_mode=color_mode,
                                                 batch_size=batch_size,
                                                 save_to_dir=save_to_dir,
                                                 save_prefix=save_prefix)

    return test_generator


class MyPreProcessor(object):
    """
    Custom preprocessing class to feed fixed rotation angle
    to ImageDataGenerator()'s apply_transform method
    """
    def __init__(self, img_shape=None, rescale=1./255, fixed_rot_angle_deg=0.):
        """
        Custom image preprocessing class
        init to implement fixed rotations.

        Parameters
        ----------
        img_shape              : list
                                 image shape (height, width, channels)
        rescale                : float
                                 rescaling value
        fixed_rot_angle_deg    : float
                                 fixed image rotation angle value
        """
        self.rescale     = rescale
        self.rot_angle   = fixed_rot_angle_deg
        self.img_shape   = img_shape
        self.datagen     = ImageDataGenerator()
        self.transform_parameters = {'theta' : fixed_rot_angle_deg}

        print('\nLoading custom preprocessing function on images of size {}'
              ' with a fixed rotation of {} deg'.format(self.img_shape, self.rot_angle))


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
        
        # Padding: Keras requires 3D tensor for 2D image
        padded_img = np.zeros((self.img_shape[0], self.img_shape[1], 1))
        proc_img = np.zeros(self.img_shape)

        # Loop through channels
        for i in range(self.img_shape[2]):
            padded_img[:,:,0] = scale_img[:,:,i].copy()
            # Rotate image channel
            rot_img = self.datagen.apply_transform(padded_img, self.transform_parameters)
            # Save for output
            proc_img[:,:,i] = rot_img[:,:,0]

        return proc_img
