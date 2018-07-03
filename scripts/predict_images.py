from __future__ import print_function

try:
    import time

    import os
    import sys
    import pickle
    import argparse
    import numpy as np

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from plots import plot_bar_probs
    from plots import plot_confusion_matrix
    from plots import plot_rotation_metrics
    from riconv.layers import Convolution2D_4
    from riconv.load_data import load_image, MyPreProcessor
    from riconv.config_utils import ModelConfigurator, DataDirStruct, ModelDirStruct

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
    parser = argparse.ArgumentParser(description="Convolutional NN Inference Script")
    parser.add_argument("-c", "--config", dest="configfile", default='config.yml', help="Path to yaml configuration file")
    parser.add_argument("-m", "--modelname", dest="modelname", type=str, required=True, help="Model name to test (only one)")
    parser.add_argument("-i", "--input", dest="input", nargs="*", required=True, help="Path to image directory or single image for inference")
    parser.add_argument("-s", "--seed", dest="rngseed", default=123, type=int, help="RNG Seed to test different samples")

    rot_parse = parser.add_mutually_exclusive_group()
    rot_parse.add_argument("-r", "--rand_rot_angle", dest="rand_rot_angle", default=0., type=float, help="Random image rotation angle range [deg]")
    rot_parse.add_argument("-f", "--fixed_rot_angle", dest="fixed_rot_angle", nargs=3, type=float, help="(low, high, spacing) fixed image rotation angle [deg]")

    args = parser.parse_args()

    # Get requested image samples
    img_list = args.input
    n_images = len(img_list)
    images = []
    shapes = []
    for img_path in img_list:
        img, shape = load_image(img_path)
        images.append(img)
        shapes.append(shape)

    # Determine which rotation to apply
    run_fixed_rotation = False
    i_results_prefix = 'random'
    rot_angle_list = [args.rand_rot_angle]
    rot_comment = "Random rotation range (deg): [-{}, {}]".format(rot_angle_list[0], 
                                                                  rot_angle_list[0])

    if args.fixed_rot_angle is not None:
        i_results_prefix = 'fixed'
        run_fixed_rotation = True
        ang_range = args.fixed_rot_angle
        rot_angle_list = np.arange(ang_range[0], ang_range[1], ang_range[2])
        rot_comment = "Fixed rotation angle(s) (deg): {}".format(rot_angle_list)

    # Get configuration file
    hconfig = ModelConfigurator(args.configfile)

    # Class names
    class_labels = hconfig.labels

    # Get requested model
    modelname = args.modelname
    print('\nTesting {} over following rotations: {} ...\n'.format(modelname, rot_angle_list))

    # Set model config parameters
    hconfig.model_config(modelname)

    # Extract model path from config
    model_dir_struct = ModelDirStruct(main_dir=hconfig.model_outpath, test_model=True)
    
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

    # Loop over images
    for iidx, test_image in enumerate(images):

        # Image name
        ibase = os.path.splitext(os.path.basename(img_list[iidx]))[0]

        # Dictionary of test results
        out_dict = {}
        out_dict['theta'] = np.array(rot_angle_list, dtype='float32')
    
        # List of probabilities for each rotation
        prob_rot_list = []
        # Run over rotation angles in list,
        # or just single value used for random range
        for i, rot_angle in enumerate(rot_angle_list):

            print('On {} angle {}'.format(i_results_prefix, rot_angle))

            # Preprocess image
            img_shape = (hconfig.height, hconfig.width, hconfig.channels)
            assert img_shape[0] <= shapes[iidx][0] and img_shape[1] <= shapes[iidx][1], \
                   "Model expected shape {} not equal to or less than loaded image shape {}" \
                   .format(img_shape, shapes[iidx])

            prep = MyPreProcessor(img_shape=img_shape,
                                  rescale=1./255,
                                  fixed_rot_angle_deg=rot_angle)
    
            proc_image = prep.preprocess_img(test_image)
            proc_image = np.expand_dims(proc_image, axis=0)

            # Choose same batch
            np.random.seed(args.rngseed)

            test_prefix = 'test_%s_rot_%.0f'%(i_results_prefix, rot_angle)

            # Predict classification
            Y_pred = trained_model.predict(proc_image)
            y_predict = np.argmax(Y_pred, axis=1)
            print('Prediction Probabilities: {}'.format(Y_pred))
            print('Class Prediction: {}'.format(y_predict))

            # Classification probability for each orientation
            prob_rot_list.append(Y_pred)

        # Transpose list
        prob_rot_arr = np.array(prob_rot_list, dtype='float32')
        class_prob_arr = prob_rot_arr.T

        # Save to dictionary
        for lidx, label in enumerate(hconfig.labels):
            class_probs = class_prob_arr[lidx][0]
            # Model's accuracies
            newkey = label + '_probability'
            out_dict[newkey] = class_probs
            print('Probabilities for class {} with model {}: {}'.format(label, modelname, class_probs))
        
        print('\nRotations and class probabilities for all')
        print_dict(out_dict)

        print('Saved some figures in {}'.format(model_dir_struct.plots_dir))

        if run_fixed_rotation: 

            rot_seq = rot_angle_list[0]
            rot_names = '%s'%rot_seq
            if len(rot_angle_list) > 1:
                rot_seq = (rot_angle_list[0], len(rot_angle_list)-2, rot_angle_list[-1])
                rot_names = '_'.join(map(str, rot_seq)).replace(" ", "")

            # Prefix
            pprefix = ibase + '_rot_' + i_results_prefix + \
                      '_test_' + rot_names

            # Save to pickel file
            pklname = pprefix + '.pkl'
            filename = os.path.join(model_dir_struct.main_dir, pklname)
            with open(filename, 'wb') as file_pi:
                pickle.dump(out_dict, file_pi)
            print("\nSaved rotation test to disk: {}\n".format(filename))

            # Plot some prediction probabilites for some rotations
            plot_bar_probs(out_dict, hconfig.labels, pprefix, model_dir_struct.plots_dir)
            # Plot rotation metrics
            plot_rotation_metrics(out_dict, ['Probability'], pprefix, model_dir_struct.plots_dir)



if __name__ == "__main__":
    main()
