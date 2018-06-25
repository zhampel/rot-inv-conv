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

def minor_eye(dim=3):
    """
    Function to provide minor diagonal of ones

    Parameters
    ----------
    dim    : int
             Sq. matrix dimension

    Returns
    -------
    matrix : array
             Output minor diagonal eye matrix
    """
    matrix = np.eye(dim, dtype='float32')
    # Run diag -> minor diag swap
    for i in range(dim):
        matrix[i][i], matrix[i][dim-i-1] = \
            matrix[i][dim-i-1], matrix[i][i]

    return matrix


def rotate_ninety(w):
    """
    Rotate kernel according to 
    requested (via shift) permutation.

    Parameters
    ----------
    w      : array_like
             Input kernel

    Returns
    -------
    w      : array_lit
             90 deg rotated kernel
    """
    shape = w.get_shape()
    m_eye = minor_eye(shape[0])

    # Right angle rotation:
    # w.T * minor_eye = R(theta = 90 deg) * w
    w_rot = tf.matmul(w, m_eye, transpose_a=True) #, b_is_sparse=True)

    return w_rot


# Convolution layer with rotated filter activations
class Convolution2D_4(Convolution2D):
    """
       Convolution2D_4 inherits from Convolution2D
       No new input variables, same output
    """
    def call(self, x, mask=None):
        # Grab the kernel(s)
        w = self.kernel
        # Make list of rotated versions
        w_rot = [w]
        for i in range(3):
            # Rotate previous kernel in list
            w = rotate_ninety(w_rot[i])
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

