import numpy as np
import tensorflow as tf

#from 'creative applications of deepLearning' Parag K Mital

#linear matrix multiplication of weights and inputs
def linear(x, n_output, name=None, activation=None, reuse=None):

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name or "fc", reuse=reuse):
        W = tf.get_variable(
                name='W',
                shape=[n_input, n_output],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(name='b',
                shape=[n_output],
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
                name='h',
                value=tf.matmul(x, W),
                bias=b)
        if activation:
            h = activation(h)

        return h, W

 
#2d convolution for generating weight-kernals and output
def conv2d(x, n_output, k_h=5, k_w=5, d_h=2, d_w=2, padding='SAME',
        name='conv2d', reuse=None):

    with tf.variable_scope(name or 'conv2d', reuse=reuse):
        W = tf.get_variable(
                name='W',
                shape=[k_h, k_w, x.get_shape()[-1], n_output],
                initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(
                name='conv',
                input=x,
                filter=W,
                strides=[1, d_h, d_w, 1],
                padding=padding)

        b = tf.get_variable(
                name='b',
                shape=[n_output],
                initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(name='h',value=conv,bias=b)

    return h, W


#deconvolution for the decoder (using tf.nn.conv2d_transpose)
def deconv2d(x, n_output_h, n_output_w, n_output_ch, n_input_ch=None,
        k_h=5, k_w=5, d_h=2, d_w=2, padding='SAME',
        name='deconv2d', reuse=None):

    with tf.variable_scope(name or 'deconv2di', reuse=reuse):
        W1 = tf.get_variable(
                name='W1',
                shape=[k_h, k_w, n_output_ch, n_input_ch or x.get_shape()[-1]],
                initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d_transpose(
                name='conv_t',
                value=x,
                filter=W1,
                output_shape=tf.stack(
                    [tf.shape(x)[0], n_output_h, n_output_w, n_output_ch]),
                strides=[1, d_h, d_w, 1],
                padding=padding)
        
        conv.set_shape([None, n_output_h, n_output_w, n_output_ch])

        b = tf.get_variable(
                name='b',
                shape=[n_output_ch],
                initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(name='h',value=conv,bias=b)

    return h, W1

#reshape tensors into one long line for comparison and addition to other flattened tensors
def flatten(x, name=None, reuse=None):

    with tf.variable_scope('flatten'):
        dims = x.get_shape().as_list()
        if len(dims) == 4:
            flattened = tf.reshape(x, shape=[-1, dims[1]*dims[2]*dims[3]])
        elif len(dims)==2 or len(dims)==1:
            flattened = x

        return flattened


def crop(img):
    return img[30:-30, 70:-70, :]
