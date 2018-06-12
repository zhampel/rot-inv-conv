from __future__ import print_function
import os.path
import numpy as np
import cv2
from skimage.transform import rotate


## Rotate image by angle given in degrees
def rotate_image(img, rot_ang_deg=0.):
    rot_img = rotate(img, rot_ang_deg)
    return rot_img

## Got to line in file
def skipton(infile, n):
    with open(infile,'r') as fi:
        for i in range(n-1):
            next(fi)
        return next(fi)

def process_img(img):

    # Normalize
    img = img.astype('float32')
    img /= 255

    height, width, channels = img.shape
    #print("Image shape: ", height, width, channels)
    return img


# Load set of images
# and rotate if provided angle
def load_img(path_to_data='', img_id=0, file_type='png'):

    # Got to line for specific id, get id and label
    file_line = skipton(path_to_data+'/labels.dat', img_id+1)
    check_id, label = file_line.split(',', )
    check_id = int(check_id)
    label = int(label)

    assert (img_id == check_id), "Image id not correctly found in file"

    # Image file string, remove extra slash if necessary
    img_file = path_to_data + '/' + str(img_id)+'_' + str(label) + "." + file_type
    img_file = img_file.replace('//','/')

    try:
        open(img_file, 'r')
    except IOError:
        print("File %s does not exist!"%img_file)
        return

    # Extract image file
    img = cv2.imread(img_file)
    img = process_img(img)
    cv2.imshow('image', img)

    return img, label


def img_generator(X, y, batch_size=32, n_classes=10, n_samples=1):
    while True:
        xs = []
        ys = []
        for _ in xrange(batch_size):
            i = np.random.randint(0, n_samples)
            y_class = np_utils.to_categorical(y[i], n_classes)
            X_data = X[i,:]
            xs.append(X_data)
            ys.append(y_class)
            #i += 1
        #print(i)
        yield(np.asarray(xs), np.asarray(ys))


#######################
# OLD CODE.... DO NOT USE
#######################



def load_data_old(path_to_data, rot_ang_deg=0.):

    # Load MNIST data set
    if (path_to_data == 'mnist'):
        x_train, y_train, x_test, y_test = load_mnist(rot_ang_deg)

    
    rotate(probability=0.5, max_left_rotation=5, max_right_rotation=10)
        
    # convert the data to the right type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    return x_train, y_train, x_test, y_test

# Specifically load hand-written digit set
def load_mnist(rot_ang_deg=0.):
    
    import keras
    from keras.datasets import mnist

    # Number of classes in set
    num_classes = 10
    
    # load the MNIST data set, which already splits into train and test sets for us
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    n_samples, img_x, img_y = x_train.shape[0], x_train.shape[1], x_train.shape[2]
    
    # reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
    # because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
    x_train = x_train.reshape(n_samples, img_x, img_y, 1)
    x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)

    if (rot_ang_deg != 0.):
        # Rotate training samples
        for j in range(n_samples):
            x_train[j,:,:,0] = rotate_image(x_train[j,:,:,0], rot_ang_deg)

        # Rotate testing samples
        for j in range(x_test.shape[0]):
            x_test[j,:,:,0] = rotate_image(x_test[j,:,:,0], rot_ang_deg)
    
    # convert class vectors to binary class matrices - this is for use in the
    # categorical_crossentropy loss below
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    return x_train, y_train, x_test, y_test
