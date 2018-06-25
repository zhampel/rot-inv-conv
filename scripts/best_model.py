from __future__ import print_function

try:
    import time

    import os
    import sys
    import pickle
    import argparse
    from shutil import copyfile

    from riconv.config_utils import ModelConfigurator, DataDirStruct, ModelDirStruct

except ImportError as e:
    print(e)
    raise ImportError

# Dictionary printer
def print_dict(dct):
    for key, value in sorted(dct.items(), reverse=True):
        print("{}: {}".format(key, value))


def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Testing Script")
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

    # Directory structures for data and model saving
    data_dir_struct = DataDirStruct(datapath)
    
    # Loop over requested models
    for mod_i in model_list:

        mod_i = mod_i.strip()

        # Set model config parameters
        hconfig.model_config(mod_i)

        # Extract model path from config
        model_dir_struct = ModelDirStruct(main_dir=hconfig.model_outpath, test_model=True)
       
        # Rename weights file
        weights_file = model_dir_struct.weights_file
        replace_name = weights_file.replace("weights", "last_epoch_weights_not_best")

        # If already replaced, then don't do it again!
        if os.path.exists(replace_name):
            continue

        # Rename weights from final iteration
        os.rename(weights_file, replace_name)

        # Get best saved weights from epochs dir, copy to weights.h5
        epoch_dir = model_dir_struct.epochs_dir
        epoch_weights = sorted(os.listdir(epoch_dir))
        best_weights = os.path.join(epoch_dir, epoch_weights[-1])

        copyfile(best_weights, weights_file)

        # Remove all but best weights
        for w_i in epoch_weights[:-1]:
            os.remove(os.path.join(epoch_dir, w_i))

        print("Moved {}'s best trained weights {} to {}.".format(mod_i, best_weights, weights_file))


if __name__ == "__main__":
    main()
