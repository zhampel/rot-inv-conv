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
    parser.add_argument("-n", "--num_samples", dest="num_samples", default=10, type=int, help="Number of test samples")

    rot_parse = parser.add_mutually_exclusive_group()
    rot_parse.add_argument("-r", "--rand_rot_angle", dest="rand_rot_angle", default=0., type=float, help="Random image rotation angle range [deg]")
    rot_parse.add_argument("-f", "--fixed_rot_angle", dest="fixed_rot_angle", nargs="*", type=float, help="Fixed image rotation angle [deg]")

    args = parser.parse_args()

    # Determine which rotation to apply
    run_fixed_rotation = False
    i_results_prefix = 'random'
    rot_angle_list = [args.rand_rot_angle]

    if args.fixed_rot_angle is not None:
        i_results_prefix = 'fixed'
        run_fixed_rotation = True
        rot_angle_list = args.fixed_rot_angle

    # Get configuration file
    with open(args.configfile, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Extract config parameters
    datapath = cfg.get('dataset', '')
    modelpath = cfg.get(args.modelname).get('outpath', 'saved_models/'+args.modelname)

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

    # Get requested sample size
    num_samples = args.num_samples

    # Run over rotation angles in list,
    # or just single value used for random range
    for i, rot_angle in enumerate(rot_angle_list):

        # Testing generator
        test_gen = test_img_generator(dir_struct=data_dir_struct, \
                                      batch_size=num_samples, \
                                      fixed_rotation=run_fixed_rotation, \
                                      rotation_angle=rot_angle, \
                                      save_to_dir=model_dir_struct.main_dir + '/figures/',
                                      save_prefix='test_%s_rot_%03i'%(i_results_prefix, i))

        test_gen.next()

    print('Saved some figures in {}'.format(model_dir_struct.main_dir + '/figures/'))


if __name__ == "__main__":
    main()
