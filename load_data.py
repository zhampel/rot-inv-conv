from __future__ import print_function
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


def train_img_generator(path_to_data='', target_size=(32, 32), batch_size=32, val_split=0.2):

    if not any(target_size):
        raise ValueError('Invalid image dimensions {}. '
                         'All elements must be >0.'.format(target_size))

    if batch_size <= 0:
        raise ValueError('Invalid batch size {}. '
                         'Must be >=0.'.format(batch_size))

    datagen = ImageDataGenerator(rescale=1./255, validation_split=val_split)

    train_generator = datagen.flow_from_directory(path_to_data,
                                                  #target_size=target_size,
                                                  batch_size=batch_size,
                                                  subset='training')

    val_generator = datagen.flow_from_directory(path_to_data,
                                                #target_size=target_size,
                                                batch_size=batch_size,
                                                subset='validation')

    return train_generator, val_generator

def test_img_generator(path_to_data='', target_size=(0, 0), batch_size=1):

    if not any(target_size):
        raise ValueError('Invalid image dimensions {}. '
                         'All elements must be >0.'.format(target_size))

    if batch_size <= 0:
        raise ValueError('Invalid batch size {}. '
                         'Must be >=0.'.format(batch_size))

    datagen = ImageDataGenerator(rescale=1./255)

    test_generator = datagen.flow_from_directory(path_to_data,
                                                 target_size=target_size,
                                                 batch_size=batch_size)

    return test_generator


def img_generator(path_to_data='', batch_size=32, n_classes=10, samples=[0,1]):
    while True:
        xs = []
        ys = []
        for _ in range(batch_size):
            i = np.random.randint(samples[0], samples[1])
            img, label = load_img(path_to_data, img_id=i, file_type='png')
            y_class = np_utils.to_categorical(label, n_classes)
            padded_img = np.zeros((img.shape[0], img.shape[1], 1))
            padded_img[:,:,0] = img
            xs.append(padded_img)
            ys.append(y_class)
        yield(np.asarray(xs), np.asarray(ys))
