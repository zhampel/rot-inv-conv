from __future__ import print_function
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
