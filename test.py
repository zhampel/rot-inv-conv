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
    from load_data import test_img_generator
    from sklearn.metrics import confusion_matrix
    from keras.models import model_from_json

except ImportError as e:
    print(e)
    raise ImportError

def main():
    global args
    p = argparse.ArgumentParser(description="Convolutional NN Training Script")
    p.add_argument("-t", "--datapath", dest="datapath", required=True, help="Directory for test image data")
    p.add_argument("-m", "--modelpath", dest="modelpath", default='saved_models', help="Saved model directory")
    p.add_argument("-n", "--num_samples", dest="num_samples", default=128, type=int, help="Number of test samples")
    args = p.parse_args()
   
    # Get number of requested batches
    num_samples = args.num_samples
    if num_samples <= 0:
        raise ValueError('Invalid number of samples {}. '
                         'Must be >0.'.format(num_samples))

    # Directory structures for data and model saving
    data_dir_struct = DataDirStruct(args.datapath)
    model_dir_struct = ModelDirStruct(args.modelpath)
    
    # Testing generator
    test_gen = test_img_generator(dir_struct=data_dir_struct, \
                                  batch_size=num_samples)

    # load json and create model
    json_file = open(model_dir_struct.model_file, 'r')
    trained_model_json = json_file.read()
    json_file.close()
    trained_model = model_from_json(trained_model_json)
    # load weights into model
    trained_model.load_weights(model_dir_struct.weights_file)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    trained_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    scores = trained_model.evaluate_generator(test_gen, max_queue_size=num_samples, steps=10)
    print("%s: %.2f%%" % (trained_model.metrics_names[1], scores[1]*100))
    y_predict = trained_model.predict_generator(test_gen, steps=num_samples)
    cm = confusion_matrix(test_gen.classes, y_predict)

if __name__ == "__main__":
    main()
