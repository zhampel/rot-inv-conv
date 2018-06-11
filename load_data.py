from __future__ import print_function
import numpy as np

def load_data(path_to_data):

    # Load MNIST data set
    if (path_to_data == 'mnist'):
        x_train, y_train, x_test, y_test = load_mnist()
        
    # Load MNIST data set
    if (path_to_data == 'mnist_rot'):
        x_train, y_train, x_test, y_test = load_mnist_rot()

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
def load_mnist():
    
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
    
    # convert class vectors to binary class matrices - this is for use in the
    # categorical_crossentropy loss below
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    return x_train, y_train, x_test, y_test

# Specifically load hand-written digit set
def load_mnist_rot():
    
    import keras
    from keras.datasets import mnist

    # Number of classes in set
    num_classes = 10
    
    # load the MNIST data set, which already splits into train and test sets for us
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    n_samples, img_x, img_y = x_train.shape[0], x_train.shape[1], x_train.shape[2]
    
    print(x_train.shape)
    for j in range(n_samples):
        x = x_train[j,:,:].copy()
        x = np.rot90(x)
        x_train[j,:,:] = x.copy()

    for j in range(x_test.shape[0]):
        x = x_test[j,:,:].copy()
        x = np.rot90(x)
        x_test[j,:,:] = x.copy()
    
    # reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
    # because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
    x_train = x_train.reshape(n_samples, img_x, img_y, 1)
    x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
    
    # convert class vectors to binary class matrices - this is for use in the
    # categorical_crossentropy loss below
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    #print(type(x_train))
    #for j in range(n_samples):
    #    x = x_train[j,:,:,0].copy()
    #    x = np.rot90(x)
    #    x_train[j,:,:,0] = x.copy()
        
    #for j in range(x_test.shape[0]):
    #    x = x_test[j,:,:,0].copy()
    #    x = np.rot90(x)
    #    x_test[j,:,:,0] = x.copy()
    
    return x_train, y_train, x_test, y_test
