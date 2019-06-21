#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import test_event_read as read
import c3d_model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = c3d_model.batch_size
num_frames = c3d_model.num_frames
height = c3d_model.height
width = c3d_model.width
channels = c3d_model.channels
n_classes = c3d_model.NUM_CLASSES
s = read.segment

model_filename = "chckPts/save40300.ckpt"


def _variable_with_weight_decay(name, shape):
    var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    return var


with tf.device('/gpu:1'):
    with tf.Graph().as_default():

        inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, num_frames, height, width, channels))

        with tf.variable_scope('var_name') as var_scope:
            weights = {
                'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64]),
                'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128]),
                'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256]),
                'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256]),
                'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512]),
                'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512]),
                'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512]),
                'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512]),
                'wd1': _variable_with_weight_decay('wd1', [8192, 4096]),
                'wd2': _variable_with_weight_decay('wd2', [4096, 4096]),
                'out': _variable_with_weight_decay('wout', [4096, n_classes])

            }

            biases = {
                'bc1': _variable_with_weight_decay('bc1', [64]),
                'bc2': _variable_with_weight_decay('bc2', [128]),
                'bc3a': _variable_with_weight_decay('bc3a', [256]),
                'bc3b': _variable_with_weight_decay('bc3b', [256]),
                'bc4a': _variable_with_weight_decay('bc4a', [512]),
                'bc4b': _variable_with_weight_decay('bc4b', [512]),
                'bc5a': _variable_with_weight_decay('bc5a', [512]),
                'bc5b': _variable_with_weight_decay('bc5b', [512]),
                'bd1': _variable_with_weight_decay('bd1', [4096]),
                'bd2': _variable_with_weight_decay('bd2', [4096]),
                'out': _variable_with_weight_decay('bout', [n_classes]),
            }

        outputs = c3d_model.inference_c3d(
            inputs_placeholder,
            0.5,
            weights,
            biases)
        outputs = tf.nn.softmax(outputs)
        predict = []

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, model_filename)

            # test
            videos = read.readFile()
            for l in videos:

                for batch in range(int(len(l[1]) / batch_size)):

                        nextX, segments = read.readTrainData(batch, l, batch_size)

                        feed_dict = {inputs_placeholder: nextX}

                        output = sess.run(outputs, feed_dict=feed_dict)
                        for i in range(batch_size):
                            p_label = np.argmax(output[i])
                            if p_label != 11:
                                p = segments[i][0] + ' ' + str(segments[i][1]) + ' ' + str(segments[i][2]) + ' ' + \
                                    str(p_label) + ' ' + str(output[i][p_label])
                                p = p+'\n'
                                write_file = open(l[0].split('.')[0] + "_" + str(s) + ".txt", "a")
                                write_file.write(p)
                                write_file.close()

