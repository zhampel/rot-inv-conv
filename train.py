from __future__ import print_function

try:
    import time

    import numpy as np
    import sys
    import os
    import argparse

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from load_data import train_img_generator
    from model import model
except ImportError as e:
    print(e)
    raise ImportError

def main():
    global args
    p = argparse.ArgumentParser(description="Convolutional NN Training Script")
    p.add_argument("-f", "--filepath", dest="filepath", required=True, help="Directory for training/testing image data")
    p.add_argument("-n", "--num_classes", dest="num_classes", default=10, type=int, help="Number of classes in data set")
    p.add_argument("-b", "--batch_size", dest="batch_size", default=128, type=int, help="Batch size")
    args = p.parse_args()
    
    filepath = args.filepath

    # Get number of requested batches
    batch_size = args.batch_size

    ### Define training and validation generators
    trainpath=filepath+'/training/'
    input_shape = (32, 32)
    train_gen, valid_gen = train_img_generator(path_to_data=trainpath, \
                                               target_size=input_shape, \
                                               batch_size=batch_size, \
                                               val_split=0.2)

    # Get Data info for model
    input_shape = train_gen.image_shape
    num_classes = train_gen.num_classes
    
    if num_classes != args.num_classes:
        raise ValueError('Expected number of classes {} '
                         'not equal to that found in data set {}.'.format(args.num_classes, num_classes))

    # Train model
    #history, trained_model = model(input_shape, num_classes,  train_gen, valid_gen)
    history, trained_model = model(train_gen, valid_gen)

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
    
    # Summarize loss history
    ax = fig.add_subplot(122)
    plt.plot(history.loss)
    plt.plot(history.val_loss)
    ax.set_title('Model Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['train', 'val'], loc='upper right')

    fig.savefig('figures/acc_loss.png')


if __name__ == "__main__":
    main()
