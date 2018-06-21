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
    from plots import plot_confusion_matrix

except ImportError as e:
    print(e)
    raise ImportError

def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-c", "--config", dest="configfile", default='config.yml', help="Path to yaml configuration file")
    parser.add_argument("-m", "--modelname", dest="modelname", required=True, help="Model name to test")

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
    with open(args.configfile, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Extract config parameters
    datapath = cfg.get('dataset', '')
    modelpath = cfg.get(args.modelname).get('outpath', os.path.join('saved_models', args.modelname))

    # Directory structures for data and model saving
    data_dir_struct = DataDirStruct(datapath)
    model_dir_struct = ModelDirStruct(modelpath)
       
    ## Load model to test
    # Load pretrained model from file
    json_file = open(model_dir_struct.model_file, 'r')
    trained_model_json = json_file.read()
    json_file.close()
    trained_model = model_from_json(trained_model_json)

    # Load weights into model
    trained_model.load_weights(model_dir_struct.weights_file)
    print("Loaded model from disk")

    # Compile trained model
    trained_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Print test results to file
    results_file = os.path.join(model_dir_struct.main_dir, 'tests.log')
    glob_text_file = open(results_file, 'w')

    glob_text_file.write('#Index\tAngle\tAccuracies\n')

    # Run over rotation angles in list,
    # or just single value used for random range
    for i, rot_angle in enumerate(rot_angle_list):

        test_prefix = 'test_%s_rot_%03i'%(i_results_prefix, i)

        # Print test results to file
        i_results_file = os.path.join(model_dir_struct.main_dir, test_prefix + '.log')
        #'/test_%s_rot_%03i.log'%(i_results_prefix, i)
        i_text_file = open(i_results_file, 'w')

        # Testing generator
        test_gen = test_img_generator(dir_struct=data_dir_struct,
                                      batch_size=1,
                                      fixed_rotation=run_fixed_rotation,
                                      rotation_angle=rot_angle)

        # Run over entire test set
        num_samples = test_gen.n

        # Truth labels for sample
        y_truth = test_gen.classes

        # Evaluate loaded model on test data
        scores = trained_model.evaluate_generator(test_gen, steps=None, verbose=1)
        print("Test %s: %.2f%%" % (trained_model.metrics_names[1], scores[1]*100))
        
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


if __name__ == "__main__":
    main()
