from __future__ import print_function

try:
    import time

    import os
    import sys
    import time
    import argparse
    import numpy as np

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from riconv.model import conv_model
    from riconv.load_data import train_img_generator, test_img_generator
    from riconv.config_utils import ModelConfigurator, DataDirStruct, ModelDirStruct

except ImportError as e:
    print(e)
    raise ImportError

def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-c", "--config", dest="configfile", default='config.yml', help="Path to yaml configuration file")
    parser.add_argument("-m", "--modelnames", dest="modelnames", nargs="*", default=None, required=False, help="Model name to test")
    parser.add_argument("-s", "--seed", dest="rngseed", default=123, type=int, help="RNG Seed to test different samples")
    args = parser.parse_args()

    # Get configuration file
    hconfig = ModelConfigurator(args.configfile)

    # Extract config parameters
    datapath = hconfig.datapath

    # Get requested models, if None, take config's list
    model_list = args.modelnames
    if model_list is None:
        model_list = hconfig.avail_models

    # Print basic info
    print("\n\n... Starting ...")
    print("\nConfiguration file: {}".format(args.configfile))
    print("\nData set location: {}\n".format(datapath))
    print("About to train {} model(s)\n".format(len(model_list)))

    # Loop over requested models
    for mod_i in model_list:
        mod_i = mod_i.strip()

        # Choose same batch
        np.random.seed(args.rngseed)

        # Set model config parameters
        hconfig.model_config(mod_i)

        # Get config file parameters
        outpath = hconfig.model_outpath
        epochs = hconfig.epochs
        layer_string_list = hconfig.layer_string_list
        # Print out model configuration
        hconfig.print_model(mod_i)

        # Directory structures for data and model saving
        data_dir_struct = DataDirStruct(datapath)
        model_dir_struct = ModelDirStruct(outpath)

        # Training and validation generators
        train_gen, valid_gen = train_img_generator(dir_struct=data_dir_struct,
                                                   config_struct=hconfig)
       
        # Time training
        start_t = time.time()

        # Train the model
        history, trained_model = conv_model(dir_struct=model_dir_struct,
                                            train_gen=train_gen,
                                            valid_gen=valid_gen,
                                            epochs=epochs,
                                            layer_string_list=layer_string_list)

        # Get training time estimate
        elapsed = time.time() - start_t
        print("Estimated time to train {}: {} sec".format(mod_i, elapsed))

        # Test the model on a subset
        print("Running model on the test set...\n")

        # Testing generator
        test_gen = test_img_generator(dir_struct=data_dir_struct,
                                      config_struct=hconfig,
                                      fixed_rotation=False,
                                      rotation_angle=0.)
        
        # Show scores for a subset
        scores = trained_model.evaluate_generator(test_gen, steps=None, verbose=1)
        print("Testing %s: %.2f%%\n" % (trained_model.metrics_names[1], scores[1]*100))

    print("\nDone training and testing.\n")

if __name__ == "__main__":
    main()
