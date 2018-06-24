import tensorflow as tf
from keras import backend as K
from keras.layers.convolutional import Convolution2D

"""
TensorFlow convolution layer incorporating
rotations of the kernel, passing along the 
maximum convolution product from the set of 
rotations as the output activation.

Inspired by the deep-learning-experiments
of github.com/raghakot
"""

permutation = [[1, 0], [0, 0], [0, 1], [2, 0], [1, 1], [0, 2], [2, 1], [2, 2], [1, 2]]


def shift_rotate(w, shift=1):
    """
    Rotate kernel according to 
    requested (via shift) permutation.

    Parameters
    ----------
    w      : array_like
             Input kernel
    shift  : int
             Requested permutation

    Returns
    -------
    output : float
             Maximum activation
    """
    print("pooping")
    shape = w.get_shape()
    for i in range(shift):
        w = tf.reshape(tf.gather_nd(w, permutation), shape)
    return w

# Convolution layer with rotated filter activations
class Convolution2D_4(Convolution2D):
    """
       Convolution2D_4 inherits from Convolution2D
       No new input variables, same output
    """
    def call(self, x, mask=None):
        # Grab the kernel(s)
        w = self.kernel
        # Make list of rotated version
        w_rot = [w]
        for i in range(3):
            w = shift_rotate(w, shift=2)
            w_rot.append(w)

        # List of activations for each rotation
        outputs = tf.stack([K.conv2d(x, w_i, strides=self.strides,
                                     padding=self.padding,
                                     data_format=self.data_format) for w_i in w_rot])

        # Choose maximal activation to pass along
        output = K.max(outputs, 0)

        # If a bias term, incorporate according to TensorFlow or Theano ordering
        if self.bias:
            if self.data_format == 'channels_first':
                output += K.reshape(self.bias, (1, self.filters, 1, 1))
            elif self.data_format == 'channels_last':
                output += K.reshape(self.bias, (1, 1, 1, self.filters))
            else:
                raise ValueError('Invalid data_format:', self.data_format)

        output = self.activation(output)

        return output

