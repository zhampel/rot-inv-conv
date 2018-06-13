from __future__ import print_function

try:
    import time

    import numpy as np
    import sys
    import os
    import argparse

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from load_data import DirStruct
    from load_data import train_img_generator, test_img_generator
    from model import model

except ImportError as e:
    print(e)
    raise ImportError

def main():
    global args
    p = argparse.ArgumentParser(description="Convolutional NN Training Script")
    p.add_argument("-f", "--filepath", dest="filepath", required=True, help="Directory for training image data")
    p.add_argument("-b", "--batch_size", dest="batch_size", default=128, type=int, help="Batch size")
    args = p.parse_args()
   
    # Directory structure
    dir_struct = DirStruct(args.filepath)

    # Get number of requested batches
    batch_size = args.batch_size
    if batch_size <= 0:
        raise ValueError('Invalid batch size {}. '
                         'Must be >=0.'.format(batch_size))

    ### Training and validation generators
    train_gen, valid_gen = train_img_generator(dir_struct=dir_struct, \
                                               batch_size=batch_size, \
                                               val_split=0.2)

    # Train model
    history, trained_model = model(train_gen, valid_gen)


    
    test_gen = test_img_generator(dir_struct=dir_struct, \
                                  batch_size=batch_size)

    scores = trained_model.evaluate_generator(test_gen, max_queue_size=batch_size, steps=10)
    print("%s: %.2f%%" % (trained_model.metrics_names[1], scores[1]*100))

    # serialize model to JSON
    model_json = trained_model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    trained_model.save_weights("model.h5")
    print("Saved model to disk")

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
