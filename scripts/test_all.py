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

    from plots import plot_confusion_matrix
    from plots import plot_rotation_metrics
    from riconv.layers import Convolution2D_4
    from riconv.load_data import test_img_generator
    from riconv.dir_utils import ModelConfigurator, DataDirStruct, ModelDirStruct

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
    parser = argparse.ArgumentParser(description="Convolutional NN Testing Script")
    parser.add_argument("-c", "--config", dest="configfile", default='config.yml', help="Path to yaml configuration file")
    parser.add_argument("-m", "--modelnames", dest="modelnames", nargs="*", default=None, required=False, help="Model name to test")

    rot_parse = parser.add_mutually_exclusive_group()
    rot_parse.add_argument("-r", "--rand_rot_angle", dest="rand_rot_angle", default=0., type=float, help="Random image rotation angle range [deg]")
    rot_parse.add_argument("-f", "--fixed_rot_angle", dest="fixed_rot_angle", nargs="*", type=float, help="Fixed image rotation angle [deg]")

    args = parser.parse_args()

    target_names = ['Planes', 'Cars', 'Birds', 'Cats', 'Deer', 'Dogs', 'Frogs', 'Horses', 'Boats', 'Trucks']

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
    hconfig = ModelConfigurator(args.configfile)

    # Extract config parameters
    datapath = hconfig.datapath
    
    # Get requested models, if None, take config's list
    model_list = args.modelnames
    if model_list is None:
        model_list = hconfig.avail_models

    # Directory structures for data and model saving
    data_dir_struct = DataDirStruct(datapath)

    # Dictionary of test results
    out_dict = {}
    out_dict['theta'] = np.array(rot_angle_list, dtype='float32')

    # List of accuracies for each model
    acc_model_list = []
    loss_model_list = []

    # Loop over requested models
    for mod_i in model_list:

        mod_i = mod_i.strip()
        print('\nTesting {} over following rotations: {} ...\n'.format(mod_i, rot_angle_list))

        # Set model config parameters
        hconfig.model_config(mod_i)

        # Extract model path from config
        modelpath = hconfig.model_outpath
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
        print("Loaded {} from disk".format(mod_i))

        # Compile trained model
        trained_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        # Print test results to file
        results_file = os.path.join(model_dir_struct.main_dir, 'tests.log')
        glob_text_file = open(results_file, 'w')

        glob_text_file.write('#Index\tAngle\tAccuracies\n')

        # List of accuracies for each rotation
        acc_rot_list = []
        loss_rot_list = []
        # Run over rotation angles in list,
        # or just single value used for random range
        for i, rot_angle in enumerate(rot_angle_list):

            print('On {} angle {}'.format(i_results_prefix, rot_angle))

            test_prefix = 'test_%s_rot_%03i'%(i_results_prefix, i)

            # Print test results to file
            i_results_file = os.path.join(model_dir_struct.main_dir, test_prefix + '.log')
            i_text_file = open(i_results_file, 'w')

            # Testing generator
            test_gen = test_img_generator(dir_struct=data_dir_struct,
                                          batch_size=1,
                                          fixed_rotation=run_fixed_rotation,
                                          rotation_angle=rot_angle)

            # Truth labels for sample
            y_truth = test_gen.classes

            # Evaluate loaded model on test data
            scores = trained_model.evaluate_generator(test_gen, steps=None, verbose=1)
            print("Test %s: %.2f%%" % (trained_model.metrics_names[1], scores[1]*100))

            # Save each rotation loss & accuracy
            loss_rot_list.append(scores[0])
            acc_rot_list.append(scores[1])
            
            # Running prediction
            Y_pred = trained_model.predict_generator(test_gen, steps=None, verbose=1)
            y_predict = np.argmax(Y_pred, axis=1)

            # Confusion matrix
            print('Confusion Matrix')
            cm = confusion_matrix(y_truth, y_predict)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print(cm)
            plot_confusion_matrix(cm=cm,
                                  classes=target_names,
                                  outname='cm_%s'%test_prefix,
                                  model_dir_struct=model_dir_struct)

            # Classification report
            print('Classification Report')
            class_report = classification_report(y_truth, 
                                                 y_predict, 
                                                 target_names=target_names)
            print(class_report)

            # Print test results to file
            i_text_file.write('\n\nRotation Angle: {} deg\n\n'.format(rot_angle))
            i_text_file.write('\n\nConfusion Matrix:\n\n')
            i_text_file.write('{}'.format(cm))

            i_text_file.write('\n\n\nClassification Report:\n\n')
            i_text_file.write('{}'.format(class_report)) 
            i_text_file.close()
            print('Saved single rotation test results to {}'.format(i_results_file))
            
            # Saving accuracy diagonals to file
            glob_text_file.write('{}\t{}\t{}'.format(i, rot_angle, cm.diagonal()).replace('[', '').replace(']', ''))

        glob_text_file.close()
        print('Saved test results to {}'.format(results_file))
        
        # Model's accuracies
        acc_model_list.append(acc_rot_list)
        loss_model_list.append(loss_rot_list)
        out_dict[mod_i+'_accuracy'] = np.array(acc_rot_list, dtype='float32')
        out_dict[mod_i+'_loss'] = np.array(loss_rot_list, dtype='float32')
        print('Accuracies for {}: {}'.format(mod_i, acc_rot_list))

    print('\nRotations, accuracies and losses for all')
    print_dict(out_dict)

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
        pprefix = 'rot_' + i_results_prefix + '_test_' + model_names + "_" + rot_names

        # Pickel file
        pklname = pprefix + '.pkl'
        filename = os.path.join(head_dir, pklname)
        with open(filename, 'wb') as file_pi:
            pickle.dump(out_dict, file_pi)
        print("\nSaved rotation test to disk: {}\n".format(filename))

        # Plot rotation metrics
        plot_rotation_metrics(out_dict, ['Accuracy','Loss'], pprefix, model_dir_struct)


if __name__ == "__main__":
    main()
