from __future__ import print_function

try:
    import time

    import os
    import sys
    import yaml
    import pickle
    import argparse
    import numpy as np

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from plots import plot_rotation_metrics
    from riconv.layers import Convolution2D_4
    from riconv.load_data import test_img_generator
    from riconv.dir_utils import DataDirStruct, ModelDirStruct

    from keras.models import model_from_json
    from sklearn.metrics import classification_report, confusion_matrix

except ImportError as e:
    print(e)
    raise ImportError

# Layer dictionary for loading json weights w/custom layer 
custom_layer_dict = {'Convolution2D_4': Convolution2D_4}

# Dictionary printer
def print_dict(dct):
    for key, value in sorted(dct.items(), reverse=True):
        print("{}: {}".format(key, value))


def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-c", "--config", dest="configfile", default='config.yml', help="Path to yaml configuration file")
    parser.add_argument("-m", "--modelnames", dest="modelnames", nargs="*", default=None, required=False, help="Model name to test")
    parser.add_argument("-n", "--num_samples", dest="num_samples", default=10, type=int, help="Number of test samples")
    parser.add_argument("-s", "--seed", dest="rngseed", default=123, type=int, help="RNG Seed to test different samples")

    rot_parse = parser.add_mutually_exclusive_group()
    rot_parse.add_argument("-r", "--rand_rot_angle", dest="rand_rot_angle", default=0., type=float, help="Random image rotation angle range [deg]")
    rot_parse.add_argument("-f", "--fixed_rot_angle", dest="fixed_rot_angle", nargs="*", type=float, help="Fixed image rotation angle [deg]")

    args = parser.parse_args()

    # Get requested sample size
    num_samples = args.num_samples

    # Determine which rotation to apply
    run_fixed_rotation = False
    i_results_prefix = 'random'
    rot_angle_list = [args.rand_rot_angle]
    rot_comment = "Random rotation range (deg): [-{}, {}]".format(rot_angle_list[0], 
                                                                  rot_angle_list[0])

    if args.fixed_rot_angle is not None:
        i_results_prefix = 'fixed'
        run_fixed_rotation = True
        rot_angle_list = args.fixed_rot_angle
        rot_comment = "Fixed rotation angle(s) (deg): {}".format(rot_angle_list)

    # Get configuration file
    with open(args.configfile, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Extract config parameters
    datapath = cfg.get('dataset', '')
    #modelpath = cfg.get(args.modelname).get('outpath', 'saved_models/'+args.modelname)

    # List of available models
    avail_models = cfg.get('models_to_run', '').split(',')
    # Get requested models, if None, get list from config
    model_list = args.modelnames
    if model_list is None:
        model_list = avail_models

    # Directory structures for data and model saving
    data_dir_struct = DataDirStruct(datapath)
    
    # Dictionary of test results
    out_dict = {}
    out_dict['theta'] = np.array(rot_angle_list, dtype='float32')

    # List of accuracies, losses, probs for each model
    acc_model_list = []
    loss_model_list = []
    prob_model_list = []

    # Loop over requested models
    for mod_i in model_list:

        mod_i = mod_i.strip()
        print('\nTesting {} over following rotations: {} ...\n'.format(mod_i, rot_angle_list))

        # Extract model path from config
        modelpath = cfg.get(mod_i).get('outpath', os.path.join(datapath, 'saved_models', mod_i))
        if not os.path.exists(modelpath):
            raise ValueError("Requested model {} has not yet been trained.".format(mod_i))

        model_dir_struct = ModelDirStruct(modelpath)
       
        ## Load model to test
        # Load pretrained model from file
        json_file = open(model_dir_struct.model_file, 'r')
        trained_model_json = json_file.read()
        json_file.close()
        trained_model = model_from_json(trained_model_json, custom_layer_dict)

        # Load weights into model
        trained_model.load_weights(model_dir_struct.weights_file)
        print("Loaded model from disk")

        # Compile trained model
        trained_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        # List of accuracies for each rotation
        prob_rot_list = []
        acc_rot_list = []
        loss_rot_list = []
        # Run over rotation angles in list,
        # or just single value used for random range
        for i, rot_angle in enumerate(rot_angle_list):

            # Choose same batch
            np.random.seed(args.rngseed)

            print('On {} angle {}'.format(i_results_prefix, rot_angle))

            # Testing generator
            test_gen = test_img_generator(dir_struct=data_dir_struct,
                                          batch_size=num_samples,
                                          fixed_rotation=run_fixed_rotation,
                                          rotation_angle=rot_angle)

            # Get Samples
            x_batch, y_batch = test_gen.next()
            # Evaluate scores
            scores = trained_model.evaluate(x_batch, y_batch, verbose=1)
            print("Test %s: %.2f%%" % (trained_model.metrics_names[1], scores[1]*100))
            # Predict classification
            Y_pred = trained_model.predict(x_batch)
            y_predict = np.argmax(Y_pred, axis=1)
            # Mean accuracy for batch
            # Save each rotation loss & accuracy
            loss_rot_list.append(scores[0])
            acc_rot_list.append(scores[1])
            # Mean classification probability for truth class
            mean_prob = np.sum(Y_pred*y_batch)/num_samples
            # Save each rotation loss & accuracy
            prob_rot_list.append(mean_prob)
        
        # Model's accuracies
        acc_model_list.append(acc_rot_list)
        loss_model_list.append(loss_rot_list)
        prob_model_list.append(prob_rot_list)
        out_dict[mod_i+'_accuracy'] = np.array(acc_rot_list, dtype='float32')
        out_dict[mod_i+'_loss'] = np.array(loss_rot_list, dtype='float32')
        out_dict[mod_i+'_probability'] = np.array(prob_rot_list, dtype='float32')
        print('Accuracies for {}: {}'.format(mod_i, prob_rot_list))
    
    print('\nRotations and accuracies for all')
    print_dict(out_dict)

    print('Saved some figures in {}'.format(model_dir_struct.plots_dir))

    if run_fixed_rotation: 
        # Save test information to pickle file
        head_dir = os.path.split(model_dir_struct.main_dir)[0]
        model_names = '_'.join(model_list).replace(" ", "")

        rot_seq = rot_angle_list[0]
        rot_names = '%s'%rot_seq
        if len(rot_angle_list) > 1:
            rot_seq = (rot_angle_list[0], len(rot_angle_list)-2, rot_angle_list[-1])
            rot_names = '_'.join(map(str, rot_seq)).replace(" ", "")

        # Prefix
        pprefix = 'rot_' + i_results_prefix + \
                  '_batch_' + str(num_samples) + '_test_' + \
                  model_names + "_" + rot_names

        # Pickel file
        pklname = pprefix + '.pkl'
        filename = os.path.join(head_dir, pklname)
        with open(filename, 'wb') as file_pi:
            pickle.dump(out_dict, file_pi)
        print("\nSaved rotation test to disk: {}\n".format(filename))

        # Plot rotation metrics
        plot_rotation_metrics(out_dict, ['Accuracy', 'Loss', 'Probability'], pprefix, model_dir_struct)



if __name__ == "__main__":
    main()
