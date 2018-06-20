from __future__ import print_function

try:
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    from matplotlib.colors import LogNorm
    from matplotlib import rcParams
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
    fig.savefig(model_dir_struct.plots_dir+'/training_accuracy.png')
    
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
    fig.savefig(model_dir_struct.plots_dir+'/training_loss.png')
