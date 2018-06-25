from __future__ import print_function

try:
    import time

    import os
    import sys
    import pickle
    import argparse
    import numpy as np

    from plots import plot_accuracy, plot_loss, compare_accuracy
    from riconv.config_utils import ModelConfigurator, ModelDirStruct

except ImportError as e:
    print(e)
    raise ImportError

def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-c", "--config", dest="configfile", default='config.yml', help="Path to yaml configuration file")
    parser.add_argument("-m", "--modelnames", dest="modelnames", nargs="*", default=None, required=False, help="Model name to test")

    args = parser.parse_args()

    # Get configuration file
    hconfig = ModelConfigurator(args.configfile)

    # Extract config parameters
    datapath = hconfig.datapath
    
    # Get requested models, if None, take config's list
    model_list = args.modelnames
    if model_list is None:
        model_list = hconfig.avail_models

    # List to store histories
    hist_list = []

    # Loop over requested models
    for mod_i in model_list:

        mod_i = mod_i.strip()
        
        # Set model config parameters
        hconfig.model_config(mod_i)

        # Extract model path from config
        modelpath = hconfig.model_outpath
        if not os.path.exists(modelpath):
            raise ValueError("Requested model {} has not yet been trained.".format(mod_i))

        # Directory structures for model
        model_dir_struct = ModelDirStruct(main_dir=outpath, test_model=True)

        with open(model_dir_struct.hist_file, 'rb') as f:
            history = pickle.load(f) 

        hist_list.append(history)

        # Visualize history
        plot_accuracy(history=history, model_dir_struct=model_dir_struct)
        plot_loss(history=history, model_dir_struct=model_dir_struct)
           
        print('Saved figures to {}'.format(model_dir_struct.plots_dir))

    # If more than one model requested, compare validation accuracy
    if len(model_list) > 1:
        compare_accuracy(names=model_list, 
                         hist_list=hist_list, 
                         model_dir_struct=model_dir_struct)


if __name__ == "__main__":
    main()
