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


def custom_image_reader(filepath, target_mode=None, target_size=None, dim_ordering=K.image_dim_ordering(), **kwargs):

    img = PIL.Image(filepath, mode="r")
    #rot_img = rotate_image(img, rot_ang_deg)
    # Using Keras data augmenting functions
    datagen = ImageDataGenerator()
    transform_parameters = {'theta' : rot_ang_deg}

    # Padding: Keras requires 3D tensor for 2D image
    padded_img = np.zeros((img.shape[0], img.shape[1], 1))
    padded_img[:,:,0] = img.copy()

    # Rotate Image
    rot_img = datagen.apply_transform(padded_img, transform_parameters)

    return rot_img

    #imgTiff = PIL.Image(filepath, mode="r")
    #imgs = []
    #for i in range(6):
    #    imgTiff.seek(i)
    #    img = np.array(imgTiff, dtype="float32")
    #    imgs.append(img)
    #imgArr = np.stack(imgs)
    #return imgArr


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
    


#####
# OLD CODE FOR TESTING
#####
        
    # For testing
    #test_generator = datagen.flow_from_directory(dir_struct.test_dir,
    #                                             target_size=target_size, 
    #                                             batch_size=batch_size,
    #                                             image_reader=customized_image_reader)

        ## Get image dimensions
        #img_shape = scale_img.shape
        ## Output image array 
        #proc_img = np.zeros(img_shape)

        ## If just one channel
        #if len(img_shape) == 2:
        #    # Padding: Keras requires 3D tensor for 2D image
        #    padded_img = np.zeros((img.shape[0], img.shape[1], 1))
        #    padded_img[:,:,0] = scale_img.copy()

        #    # Rotate Image
        #    rot_img = self.datagen.apply_transform(padded_img, self.transform_parameters)
        #    proc_img = rot_img[:,:,0]

        ## Otherwise RGB channels
        #elif len(img_shape) == 3:
        #    padded_img = np.zeros((img.shape[0], img.shape[1], 1))
        #    # Loop through channels
        #    for i in range(img_shape[2]):
        #        padded_img[:,:,0] = scale_img[:,:,i].copy()
        #        # Rotate image channel
        #        rot_img = self.datagen.apply_transform(padded_img, self.transform_parameters)
        #        # Save for output
        #        proc_img[:,:,i] = rot_img[:,:,0]
        #return proc_img


#def custom_image_reader(filepath, target_mode=None, target_size=None, dim_ordering=K.image_dim_ordering(), **kwargs):
# https://github.com/keras-team/keras/issues/3416
#
#    img = PIL.Image(filepath, mode="r")
#    #rot_img = rotate_image(img, rot_ang_deg)
#    # Using Keras data augmenting functions
#    datagen = ImageDataGenerator()
#    transform_parameters = {'theta' : rot_ang_deg}
#
#    # Padding: Keras requires 3D tensor for 2D image
#    padded_img = np.zeros((img.shape[0], img.shape[1], 1))
#    padded_img[:,:,0] = img.copy()
#
#    # Rotate Image
#    rot_img = datagen.apply_transform(padded_img, transform_parameters)
#
#    return rot_img
        
        
        
        
        #self.rot_image = self.n_channel_rot

        ## Choose rotation function based on # image channels
        #if (self.img_shape[2] > 1):
        #    print("N Channel rotation")
        #    self.rot_image = self.n_channel_rot
        #else:
        #    print("One Channel rotation")
        #    self.rot_image = self.one_channel_rot
    
    #def one_channel_rot(self, img):
    #    """Function for single channel (black & white) image"""
    #    # Padding: Keras requires 3D tensor for 2D image
    #    padded_img = np.zeros((self.img_shape[0], self.img_shape[1], 1))
    #    padded_img[:,:,0] = img.copy()

    #    # Rotate Image
    #    rot_img = self.datagen.apply_transform(padded_img, self.transform_parameters)
    #    return rot_img[:,:,0]

    #def n_channel_rot(self, img):
    #    """Function for multiple channel (e.g. rgb) image"""
    #    # Padding: Keras requires 3D tensor for 2D image
    #    padded_img = np.zeros((self.img_shape[0], self.img_shape[1], 1))
    #    proc_img = np.zeros(self.img_shape)

    #    # Loop through channels
    #    for i in range(self.img_shape[2]):
    #        padded_img[:,:,0] = img[:,:,i].copy()
    #        # Rotate image channel
    #        rot_img = self.datagen.apply_transform(padded_img, self.transform_parameters)
    #        # Save for output
    #        proc_img[:,:,i] = rot_img[:,:,0]

    #    return proc_img

