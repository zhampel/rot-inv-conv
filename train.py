
try:

    import time

    import numpy as np
    import sys
    import os
    import argparse


    import matplotlib as mpl
    import matplotlib.pyplot as plt

    #import keras
    #from keras.layers import Dense, Flatten
    #from keras.layers import Conv2D, MaxPooling2D
    #from keras.models import Sequential
    from load_data import load_img, img_generator
    from model import model
except ImportError as e:
    print(e)
    raise ImportError

# Specifics
batch_size = 128

def main():
    global args
    p = argparse.ArgumentParser(description="Convolutional NN Training Script")
    p.add_argument("-o", "--filepath", dest="filepath", required=True, help="Directory for training/testing image data")
    p.add_argument("-n", "--num_classes", dest="num_classes", default=10, type=int, help="Number of classes in data set")
    args = p.parse_args()
    
    filepath = args.filepath

    # Data
    ## Extract single image to get info
    img, label = load_img(filepath+'/training/', img_id=1, file_type='png')
    input_shape = (img.shape[0], img.shape[1], 1)
    num_classes = args.num_classes

    print(input_shape)
    print(num_classes)

    #train_gen = img_generator(filepath, batch_size=128, \
    #                          n_classes=num_classes, n_samples=100*batch_size)
    #
    #valid_gen = img_generator(filepath, batch_size=128, \
    #                          n_classes=num_classes, n_samples=100*batch_size)

    ## Define training and validation generators
    trainpath=filepath+'/training/'
    train_gen = img_generator(trainpath, batch_size=128, n_classes=num_classes, samples=[0, 50000])
    
    valpath=filepath+'/training/'
    valid_gen = img_generator(valpath, batch_size=128, n_classes=num_classes, samples=[50000, 60000])

    # Train model
    history, trained_model = model(input_shape, num_classes, train_gen, valid_gen)

    # Summarize history
    fig = plt.figure(figsize=(14,5))
    
    # Summarize accuracy history
    ax = fig.add_subplot(121)
    ax.plot(history.acc)
    ax.plot(history.val_acc)
    ax.set_title('Model Accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(['train', 'val'], loc='upper left')
    
    ax = fig.add_subplot(122)
    # Summarize loss history
    plt.plot(history.loss)
    plt.plot(history.val_loss)
    ax.set_title('Model Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['train', 'val'], loc='upper right')

    fig.figsave('figures/acc_loss.png')



if __name__ == "__main__":
    main()
