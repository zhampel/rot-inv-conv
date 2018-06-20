from __future__ import print_function

try:
    import time

    import os
    import sys
    import yaml
    import pickle
    import argparse
    import numpy as np
    from plots import plot_accuracy, plot_loss

    from dir_utils import ModelDirStruct

except ImportError as e:
    print(e)
    raise ImportError

def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-c", "--config", dest="configfile", default='config.yml', help="Path to yaml configuration file")
    parser.add_argument("-m", "--modelname", dest="modelname", required=True, help="Model name to test")

    args = parser.parse_args()

    # Get configuration file
    with open(args.configfile, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Extract config parameters
    modelpath = cfg.get(args.modelname).get('outpath', 'saved_models/'+args.modelname)

    # Directory structures for model
    model_dir_struct = ModelDirStruct(modelpath)

    with open(model_dir_struct.hist_file, 'rb') as f:
        history = pickle.load(f) 

    # Visualize history
    plot_accuracy(history=history, model_dir_struct=model_dir_struct)
    plot_loss(history=history, model_dir_struct=model_dir_struct)
       
    print('Saved figures to {}'.format(model_dir_struct.plots_dir))

if __name__ == "__main__":
    main()
