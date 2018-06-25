from __future__ import print_function

try:
    import os
    import itertools
    import numpy as np
    from scipy.interpolate import interp1d

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import ScalarFormatter
except ImportError as e:
    print(e)
    raise ImportError

# rcParam adjustments
mpl.rc("font", family="serif", size=18)
rcParams.update({'figure.autolayout': True})
rcParams.update({'font.size': 20})
rcParams.update({'axes.titlesize':18})
rcParams.update({'axes.labelsize':16})
rcParams.update({'xtick.labelsize':15})
rcParams.update({'ytick.labelsize':15})
rcParams.update({'legend.fontsize':14})
rcParams.update({'legend.numpoints':1})
rcParams.update({'grid.linewidth':0.2})

# Colorbar options
cbarfontsize = 14
cbarticksize = 0.5

# Marker/Line Options
colors = ["blue", "red", "green", "black"]
colorsmall = ["b", "r", "g", "k"]
styles = ["-", "--", "-."]



def plot_image(img, cmap='gray', label=''):
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(img, cmap=cmap)
    plt.show()


def plot_accuracy(history=None, model_dir_struct=None):

    n_epochs = len(history['acc'])
    epochs = range(1, n_epochs+1)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(epochs, history['acc'])
    ax.plot(epochs, history['val_acc'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    plt.legend(['Train', 'Valid'], loc='upper left')
    fig.savefig(os.path.join(model_dir_struct.plots_dir, 'training_accuracy.png'))

    
def plot_loss(history=None, model_dir_struct=None):

    n_epochs = len(history['loss'])
    epochs = range(1, n_epochs+1)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(epochs, history['loss'])
    ax.plot(epochs, history['val_loss'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.legend(['Train', 'Valid'], loc='upper right')
    fig.savefig(os.path.join(model_dir_struct.plots_dir, 'training_loss.png'))


def plot_confusion_matrix(cm=None, classes=None,
                          outname='confusion_matrix', 
                          model_dir_struct=None):
    """
    This function plots the confusion matrix.
    """

    cmap = plt.cm.Blues
    title='Confusion Matrix'
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    cax = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    cbar = fig.colorbar(cax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 fontsize=8, horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    fig.savefig(os.path.join(model_dir_struct.plots_dir, outname + '.png'))


def compare_accuracy(names=None, hist_list=None, model_dir_struct=None):

    head_dir = os.path.split(model_dir_struct.main_dir)[0]
    model_names = '_'.join(names).replace(" ", "")
    filename = 'val_accuracy_' + model_names + '.png'
    figname = os.path.join(head_dir, filename)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)

    for i, hist in enumerate(hist_list):

        n_epochs = len(hist['acc'])
        epochs = range(1, n_epochs+1)

        ax.plot(epochs, hist['val_acc'], label=names[i].strip())

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    plt.legend(loc='lower right', numpoints=1)
    plt.tight_layout()
    fig.savefig(figname)
        
    print('Saved figure  to {}'.format(figname))


def plot_rotation_metrics(data_dict=None, metrics=None, prefix='', out_path=''):

    # Get rotation angles
    theta_vals = data_dict['theta']
    data_dict.pop('theta', None)

    # Plot metrics for all models' rotations
    for met in metrics: 
        filename = prefix + '_' + met + '.png'
        figname = os.path.join(out_path, filename)

        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)

        for key, met_vals in sorted(data_dict.items()):
            if (met.lower() not in key):
                continue

            # Plot data
            pp = ax.plot(theta_vals, met_vals, 
                         label=key.replace('_'+met.lower(),""), 
                         linestyle='None', marker='o', markersize=4)

            # Need min points to interpolate
            if len(theta_vals) > 3:
                tfine = np.linspace(theta_vals[0], theta_vals[-1], len(theta_vals)*100, endpoint=True)
                f2 = interp1d(theta_vals, met_vals, kind='cubic')

                ax.plot(tfine, f2(tfine), 
                        color=pp[0].get_color(), 
                        linestyle='-', linewidth=0.5)

        ax.set_xlabel('Rotation Angle [deg]')
        ax.set_ylabel('{}'.format(met))
        plt.legend(loc='best', numpoints=1)
        plt.tight_layout()
        fig.savefig(figname)
            
        print('Saved figure to {}'.format(figname))

