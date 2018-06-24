from __future__ import print_function

import os
import sys
import yaml
import os.path


class ModelConfigurator(object):
    """
    Configuration file data
    """

    # Fields, subfields required in configuration file
    head_needed  = set(["dataset", "classes", "height", "width", "channels"])
    model_needed = set(["layers"])

    def __init__(self, config_file=""):
        
        # Get configuration file
        self.filepath = os.path.abspath(config_file)
        with open(config_file, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)

        # Loaded config object
        self.cfg = cfg

        # Ensure necessary header fields exist
        if not self.check_fields(cfg=cfg, tset=self.head_needed):
            raise AssertionError("Some fields in {} not found. "
                                 "Required fields: {}".format(self.filepath, 
                                                              self.head_needed))

        # Extract config parameters
        self.datapath = cfg.get('dataset', '')

        # List of available models
        self.avail_models = cfg.get('models_to_run', '').split(',')
        self.head_outpath = cfg.get('outpath', os.path.join(self.datapath, 'saved_models'))
        self.classes  = int(cfg.get('classes'))
        self.height   = int(cfg.get('height'))
        self.width    = int(cfg.get('width'))
        self.channels = int(cfg.get('channels'))

    def model_config(self, model_name=""):
        # Get config file parameters for specific model
        self.model_name = model_name
        self.mod_cfg = self.cfg.get(model_name)

        # Ensure necessary subfields exist
        if not self.check_fields(cfg=self.mod_cfg, tset=self.model_needed):
            raise AssertionError("Some sub-fields of {} in {} not found. "
                                 "Required fields: {}".format(self.model_name, 
                                                              self.filepath, 
                                                              self.model_needed))

        self.model_outpath = os.path.join(self.head_outpath, model_name)
        self.val_split = self.mod_cfg.get('validation_split', 0.2)
        self.batch_size = self.mod_cfg.get('batch_size', 128)
        self.epochs = self.mod_cfg.get('epochs', -1)
        self.rotation_range = self.mod_cfg.get('rotation_range', 0.)
        
        layer_string_list = self.mod_cfg.get('layers', 'conv2d, conv2d, conv2d, conv2d')
        self.layer_string_list = [lay.strip() for lay in layer_string_list.split(',')]

    # Ensure basic, necessary fields are in the config file
    def check_fields(self, cfg=None, tset=None):
        seen = set()
        for key, value in cfg.items():
            seen.add(key)
        
        return tset.issubset(seen)
        
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

