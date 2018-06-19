from __future__ import print_function

try:
    import time

    import os
    import sys
    import yaml
    import argparse
    import numpy as np

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from load_data import test_img_generator
    from keras.models import model_from_json
    from sklearn.metrics import classification_report, confusion_matrix
    from dir_utils import DataDirStruct, ModelDirStruct

except ImportError as e:
    print(e)
    raise ImportError

def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-c", "--config", dest="configfile", default='config.yml', help="Path to yaml configuration file")
    parser.add_argument("-m", "--modelname", dest="modelname", required=True, help="Model name to test")
    parser.add_argument("-n", "--num_samples", dest="num_samples", default=128, type=int, help="Number of test samples")
    args = parser.parse_args()
   
    # Get configuration file
    with open(args.configfile, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Extract config parameters
    datapath = cfg.get('dataset', '')
    modelpath = cfg.get(args.modelname).get('outpath', 'saved_models/'+args.modelname)

    # Directory structures for data and model saving
    data_dir_struct = DataDirStruct(datapath)
    model_dir_struct = ModelDirStruct(modelpath)
   
    # Get requested sample size
    num_samples = args.num_samples

    # Testing generator
    test_gen = test_img_generator(dir_struct=data_dir_struct, \
                                  batch_size=num_samples)

    ## Run over entire test set
    ## unless a smaller batch is requested
    #num_samples = test_gen.n
    #if args.num_samples > 0:
    #    num_samples = args.num_samples 

    # Load model from files
    json_file = open(model_dir_struct.model_file, 'r')
    trained_model_json = json_file.read()
    json_file.close()
    trained_model = model_from_json(trained_model_json)
    # Load weights into model
    trained_model.load_weights(model_dir_struct.weights_file)
    print("Loaded model from disk")

    # Evaluate loaded model on test data
    trained_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    scores = trained_model.evaluate_generator(test_gen, max_queue_size=num_samples, steps=10)
    print("Test %s: %.2f%%" % (trained_model.metrics_names[1], scores[1]*100))
    
    # Running prediction
    Y_pred = trained_model.predict_generator(test_gen, test_gen.n // num_samples + 1)
    y_predict = np.argmax(Y_pred, axis=1)

    # Confusion matrix
    print('Confusion Matrix')
    cm = confusion_matrix(test_gen.classes[0:len(y_predict)], y_predict)
    print(cm)

    # Classification report
    print('Classification Report')
    target_names = ['Cats', 'Dogs', 'Horse', 'poop', 't','a','b','c','x','n']
    print(classification_report(test_gen.classes[0:len(y_predict)], y_predict, target_names=target_names))

if __name__ == "__main__":
    main()
