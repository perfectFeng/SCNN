#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

NUM_CLASSES = 12
height = 224
width = 224
channels = 3
num_frames = 32
batch_size = 3


def f_c3d(_input_data, _dropout, _weights, _biases):
    conv1 = tf.nn.conv3d(_input_data, _weights['wc1'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv1')
    conv1 = tf.nn.bias_add(conv1, _biases['bc1'])
    conv1 = tf.nn.relu(conv1, 'relu1')
    pool1 = tf.nn.max_pool3d(conv1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1')

    # Convolution Layer
    conv2 = tf.nn.conv3d(pool1, _weights['wc2'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv2')
    conv2 = tf.nn.bias_add(conv2, _biases['bc2'])
    conv2 = tf.nn.relu(conv2, 'relu2')
    # pooling layer
    pool2 = tf.nn.max_pool3d(conv2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool2')

    # Convolution Layer
    conv3 = tf.nn.conv3d(pool2, _weights['wc3a'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv3a')
    conv3 = tf.nn.bias_add(conv3, _biases['bc3a'])
    conv3 = tf.nn.relu(conv3, 'relu3a')
    conv3 = tf.nn.conv3d(conv3, _weights['wc3b'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv3b')
    conv3 = tf.nn.bias_add(conv3, _biases['bc3b'])
    conv3 = tf.nn.relu(conv3, 'relu3b')
    # pooling layer
    pool3 = tf.nn.max_pool3d(conv3, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool3')

    # Convolution Layer
    conv4 = tf.nn.conv3d(pool3, _weights['wc4a'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv4a')
    conv4 = tf.nn.bias_add(conv4, _biases['bc4a'])
    conv4 = tf.nn.relu(conv4, 'relu4a')
    conv4 = tf.nn.conv3d(conv4, _weights['wc4b'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv4b')
    conv4 = tf.nn.bias_add(conv4, _biases['bc4b'])
    conv4 = tf.nn.relu(conv4, 'relu4b')
    # pooling layer
    pool4 = tf.nn.max_pool3d(conv4, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool4')

    # Convolution Layer
    conv5 = tf.nn.conv3d(pool4, _weights['wc5a'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv5a')
    conv5 = tf.nn.bias_add(conv5, _biases['bc5a'])
    conv5 = tf.nn.relu(conv5, 'relu5a')
    conv5 = tf.nn.conv3d(conv5, _weights['wc5b'], strides=[1, 1, 1, 1, 1], padding='SAME', name='conv5b')
    conv5 = tf.nn.bias_add(conv5, _biases['bc5b'])
    conv5 = tf.nn.relu(conv5, 'relu5b')

    # pooling layer
    pool5 = tf.nn.max_pool3d(conv5, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool5')

    return pool5


def inference_c3d(f_data, _dropout, f_weights, f_biases):

    pool5 = f_c3d(f_data, _dropout, f_weights, f_biases)
    pool5 = tf.nn.max_pool3d(pool5, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding='SAME')

    pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3])
    dense1 = tf.reshape(pool5, [batch_size, f_weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.matmul(dense1, f_weights['wd1']) + f_biases['bd1']

    dense1 = tf.nn.relu(dense1, name='fc1')
    dense1 = tf.nn.dropout(dense1, _dropout)

    dense2 = tf.nn.relu(tf.matmul(dense1, f_weights['wd2']) + f_biases['bd2'], name='fc2')
    dense2 = tf.nn.dropout(dense2, _dropout)

    out = tf.matmul(dense2, f_weights['out']) + f_biases['out']
    return out
