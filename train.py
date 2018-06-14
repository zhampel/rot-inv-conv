from __future__ import print_function

try:
    import time

    import numpy as np
    import sys
    import os
    import argparse

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from dir_utils import DataDirStruct, ModelDirStruct
    from load_data import train_img_generator, test_img_generator
    from model import model

except ImportError as e:
    print(e)
    raise ImportError

def main():
    global args
    p = argparse.ArgumentParser(description="Convolutional NN Training Script")
    p.add_argument("-t", "--trainpath", dest="trainpath", required=True, help="Directory for training image data")
    p.add_argument("-o", "--outpath", dest="outpath", default='saved_models', help="Directory for saving trained model")
    p.add_argument("-b", "--batch_size", dest="batch_size", default=128, type=int, help="Batch size")
    p.add_argument("-v", "--val_split", dest="val_split", default=0.2, type=float, help="Validation split")
    args = p.parse_args()
   
    # Get number of requested batches
    batch_size = args.batch_size
    if batch_size <= 0:
        raise ValueError('Invalid batch size {}. '
                         'Must be >0.'.format(batch_size))

    # Get validation split fraction
    val_split = args.val_split
    if val_split <= 0 or val_split >1.:
        raise ValueError('Invalid validation split {}. '
                         'Must be between 0 and 1.'.format(val_split))

    # Directory structures for data and model saving
    data_dir_struct = DataDirStruct(args.trainpath)
    model_dir_struct = ModelDirStruct(args.outpath)

    # Training and validation generators
    train_gen, valid_gen = train_img_generator(dir_struct=data_dir_struct, \
                                               batch_size=batch_size, \
                                               val_split=val_split)
    
    # Train the model
    history, trained_model = model(dir_struct=model_dir_struct, \
                                   train_gen=train_gen, \
                                   valid_gen=valid_gen)
   
    # Testing generator
    test_gen = test_img_generator(dir_struct=data_dir_struct, \
                                  batch_size=batch_size)

    scores = trained_model.evaluate_generator(test_gen, max_queue_size=batch_size, steps=10)
    print("%s: %.2f%%" % (trained_model.metrics_names[1], scores[1]*100))

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

    fig.savefig(model_dir_struct.plots_dir+'/acc_loss.png')


if __name__ == "__main__":
    main()
