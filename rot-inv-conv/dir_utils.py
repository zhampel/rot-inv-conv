from __future__ import print_function
import sys
import os.path


class ModelDirStruct(object):
    """
    Directory structure for saving models
    """
    def __init__(self, main_dir=""):
        self.main_dir = main_dir
        self.log_file = main_dir+'/training.log'
        self.hist_file = main_dir+'/history.pkl'
        self.tb_log_file = main_dir+'/tb_log.log'
        self.model_file = main_dir+'/model.json'
        self.weights_file = main_dir+'/weights.h5'
        self.epochs_dir = main_dir+'/epochs'
        self.plots_dir = main_dir+'/figures'
        self.setup_dirs()

    def setup_dirs(self):
        if not os.path.exists(self.main_dir):
            os.mkdir(self.main_dir)
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
        self.main_dir = main_dir
        self.data_file = main_dir+'/image_data.dat'
        self.train_dir = main_dir+'/training'
        self.test_dir = main_dir+'/testing'
        self.check_dirs()

    def check_dirs(self):
        if not os.path.exists(self.main_dir):
            print('No such directory {} '\
            'does not exist!'.format(self.main_dir), file=sys.stderr)
            sys.exit()

        if not os.path.exists(self.train_dir):
            print('No such directory {} '\
            'does not exist!'.format(self.train_dir), file=sys.stderr)
            sys.exit()

        if not os.path.exists(self.test_dir):
            print('No such directory {} '\
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
