from __future__ import print_function

import os
import sys
import yaml
import os.path


class Configurator(object):
    """
    Configuration file data
    """
    def __init__(self, config_file=""):
        
        # Get configuration file
        with open(config_file, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)

        # Loaded config object
        self.cfg = cfg

        # Extract config parameters
        self.trainpath = cfg.get('dataset', '')

        # List of available models
        self.avail_models = cfg.get('models_to_run', '').split(',')
        self.head_outpath = cfg.get('outpath', os.path.join(self.trainpath, 'saved_models'))

    def model_config(self, model_name=""):
        # Get config file parameters for specific model
        self.name = model_name
        self.mod_cfg = self.cfg.get(model_name)
        self.model_outpath = os.path.join(self.head_outpath, model_name)
        self.val_split = mod_cfg.get('validation_split', 0.2)
        self.batch_size = mod_cfg.get('batch_size', 128)
        self.epochs = mod_cfg.get('epochs', -1)
        self.rotation_range = mod_cfg.get('rotation_range', 0.)
        
        layer_string_list = mod_cfg.get('layers', 'conv2d, conv2d, conv2d, conv2d')
        self.layer_string_list = [lay.strip() for lay in layer_string_list.split(',')]

    def print_model(self, model_name=""):
        print("Model {} details:\n\t{}\n".format(model_name, self.cfg.get(model_name)))
        


class ModelDirStruct(object):
    """
    Directory structure for saving models
    """
    def __init__(self, main_dir=""):
        self.main_dir     = main_dir
        self.epochs_dir   = os.path.join(main_dir, 'epochs')
        self.plots_dir    = os.path.join(main_dir, 'figures')
        self.log_file     = os.path.join(main_dir, 'training.log')
        self.hist_file    = os.path.join(main_dir, 'history.pkl')
        self.tb_log_file  = os.path.join(main_dir, 'tb_log.log')
        self.model_file   = os.path.join(main_dir, 'model.json')
        self.weights_file = os.path.join(main_dir, 'weights.h5')
        self.setup_dirs()

    def setup_dirs(self):
        if not os.path.exists(self.main_dir):
            os.makedirs(self.main_dir)
            print('Making models directory {} '.format(self.main_dir))
        if not os.path.exists(self.epochs_dir):
            os.mkdir(self.epochs_dir)
            print('Making epochs directory {} '.format(self.epochs_dir))
        if not os.path.exists(self.plots_dir):
            os.mkdir(self.plots_dir)
            print('Making plots directory {} '.format(self.plots_dir))




class DataDirStruct(object):
    """
    Expected directory structure
    for accessing image data sets
    """
    def __init__(self, main_dir=""):
        self.main_dir  = main_dir
        self.train_dir = os.path.join(main_dir, 'training')
        self.test_dir  = os.path.join(main_dir, 'testing')
        self.data_file = os.path.join(main_dir, 'image_data.dat')
        self.check_dirs()

    def check_dirs(self):
        if not os.path.exists(self.main_dir):
            print('No such directory {} '
            'does not exist!'.format(self.main_dir), file=sys.stderr)
            sys.exit()

        if not os.path.exists(self.train_dir):
            print('No such directory {} '
            'does not exist!'.format(self.train_dir), file=sys.stderr)
            sys.exit()

        if not os.path.exists(self.test_dir):
            print('No such directory {} '
            'does not exist!'.format(self.test_dir), file=sys.stderr)
            sys.exit()

        try:
            open(self.data_file, 'r')
        except Exception:
            print('File %s does not exist!'%self.data_file)


    def get_img_data(self):

        f = open(self.data_file, 'r')
        line = f.readline()
        n, h, w, c = line.split(',')
        num_classes, height, width, channels = int(n), int(h), int(w), int(c)
        print('Classes: {}, '
              'Image Dims: ({}, {}), '
              '# Channels: {}'.format(num_classes, height, width, channels))
        f.close()

        return num_classes, height, width, channels
