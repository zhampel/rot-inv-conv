import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers.convolutional import Convolution2D

"""
TensorFlow convolution layer incorporating
rotations of the kernel, passing along the 
maximum convolution product from the set of 
rotations as the output activation.
"""


"""
Permutation method inspired by the 
deep-learning-experiments of github.com/raghakot
"""
# Clockwise permutation matrix of 3x3 matrix indices
permutation = [[1, 0], [0, 0], [0, 1], 
               [2, 0], [1, 1], [0, 2], 
               [2, 1], [2, 2], [1, 2]]

def shift_rotate(w, shift=1):
    """
    Rotate 3 x 3 x n x n kernel according 
    to requested (via shift) permutation.

    Parameters
    ----------
    w      : 3 x 3 x N x N tensor
             Input kernel
    shift  : int
             Requested permutation

    Returns
    -------
    w      : tensor
             Rotated kernel
    """
    shape = w.get_shape()
    for i in range(shift):
        w = tf.reshape(tf.gather_nd(w, permutation), shape)
    return w


"""
My method: transpose of kernel, followed
by multiplication with anti-diagonal identity.
"""
def minor_eye(shape=None):
    """
    Function to provide minor diagonal of ones

    Parameters
    ----------
    dim    : tuple (int)
             Conv layer shape (kernel, filters)

    Returns
    -------
    matrix : tensor (float32)
             Output minor diagonal eye matrix
    """
    matrix = np.zeros(shape=shape, dtype='float32')
    dim = shape[0]
    for i in range(dim):
        matrix[i,dim-i-1,:,:] = 1.
    tf_matrix = tf.convert_to_tensor(matrix, np.float32)

    return tf_matrix


def rotate_ninety(w):
    """
    Rotate arbitrary square kernel according 
    to requested (via shift) permutation.

    Parameters
    ----------
    w      : k x k x n x n tensor
             Input kernel

    Returns
    -------
    w      : tensor
             90 deg rotated kernel
    """
    shape = w.get_shape()
    m_eye = minor_eye(shape)

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

        # Rotate previous kernel in list
        for i in range(3):

            # Transpose + anti-diag I
            w = rotate_ninety(w)

            # Permutation of indices
            #w = shift_rotate(w, shift=2)

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

