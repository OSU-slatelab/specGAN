from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
from os import path
import json
import glob
import random
import collections
import math
import time
from data_io import read_kaldi_ark_from_scp, kaldi_write_mats
from six.moves import xrange 
data_base_dir = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument("--noisy_train_file", default="data-spectrogram/train_si84_noisy/feats.scp", help="The input feature file for training")
parser.add_argument("--noisy_dev_file", default="data-spectrogram/dev_dt_05_noisy/feats.scp", help="The input feature file for cross-validation")
parser.add_argument("--buffer_size", default=1, type=int)
parser.add_argument("--batch_size", default=1, type=int)
#Model
parser.add_argument("--nlayers", type=int, default=2)
parser.add_argument("--gen_units", type=int, default=2048)
parser.add_argument("--disc_units", type=int, default=1024)
parser.add_argument("--input_featdim", type=int, default=257*3)
parser.add_argument("--output_featdim", type=int, default=257)
parser.add_argument("--context", type=int, default=5)
parser.add_argument("--epsilon", type=float, default=1e-3, help="parameter for batch normalization")
parser.add_argument("--decay", type=float, default=0.999, help="parameter for batch normalization")
parser.add_argument("--num_steps_per_decay", type=int, default=5312*10, help="number of steps after learning rate decay")
parser.add_argument("--decay_rate", type=float, default=0.96, help="learning rate decay")
parser.add_argument("--max_global_norm", type=float, default=5.0, help="global max norm for clipping")
parser.add_argument("--keep_prob", type=float, default=0.4, help="keep percentage of neurons")
parser.add_argument("--patience", type=int, default=5312*10, help="patience interval to keep track of improvements")
parser.add_argument("--patience_increase", type=int, default=2, help="increase patience interval on improvement")
parser.add_argument("--improvement_threshold", type=float, default=0.995, help="keep track of validation error improvement")

a = parser.parse_args()


def read_mats(uid, offset, file_name):
    #Read a buffer containing buffer_size*batch_size+offset 
    #Returns a line number of the scp file
    scp_fn = path.join(data_base_dir, file_name)
    ark_dict,uid = read_kaldi_ark_from_scp(uid, offset, a.batch_size, a.buffer_size, scp_fn, data_base_dir)
    return ark_dict,uid

def batch_norm(x, shape, training):
    '''Assume 2d [batch, values] tensor'''

    scale = tf.get_variable('scale', shape[-1], initializer=tf.constant_initializer(0.1))
    offset = tf.get_variable('offset', shape[-1])

    pop_mean = tf.get_variable('pop_mean',
                               shape[-1],
                               initializer=tf.constant_initializer(0.0),
                               trainable=False)
    pop_var = tf.get_variable('pop_var',
                              shape[-1],
                              initializer=tf.constant_initializer(1.0),
                              trainable=False)
    batch_mean, batch_var = tf.nn.moments(x, [0])

    train_mean_op = tf.assign(pop_mean, pop_mean * a.decay + batch_mean * (1 - a.decay))
    train_var_op = tf.assign(pop_var, pop_var * a.decay + batch_var * (1 - a.decay))

    def batch_statistics():
        with tf.control_dependencies([train_mean_op, train_var_op]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, a.epsilon)

    def population_statistics():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, a.epsilon)

    return tf.cond(training, batch_statistics, population_statistics)


def create_generator(generator_inputs):

    #Create additional placeholder inputs
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    # Hidden 1
    with tf.variable_scope('hidden1'):
        shape = [a.input_featdim*(2*a.context+1), a.gen_units]
        weight = tf.get_variable("weight", shape, dtype=tf.float32, initializer = tf.random_normal_initializer(0,1))
        bias = tf.get_variable("bias", shape[-1])
        linear = tf.matmul(generator_inputs, weight) + bias
        bn = batch_norm(linear, shape, is_training)
        hidden = tf.nn.relu(bn)
        dropout1 = tf.nn.dropout(hidden, keep_prob)
    # Hidden 2
    with tf.variable_scope('hidden2'):
        shape = [a.gen_units, a.gen_units]
        weight = tf.get_variable("weight", shape, dtype=tf.float32, initializer = tf.random_normal_initializer(0,1))
        bias = tf.get_variable("bias", shape[-1])
        linear = tf.matmul(dropout1, weight) + bias
        bn = batch_norm(linear, shape, is_training)
        hidden = tf.nn.relu(bn)
        dropout2 = tf.nn.dropout(hidden, keep_prob)
    # Linear
    with tf.variable_scope('linear'):
        shape = [a.gen_units, a.output_featdim]
        weight = tf.get_variable("weight", shape, dtype=tf.float32, initializer = tf.random_normal_initializer(0,1))
        bias = tf.get_variable("bias", shape[-1])
        linear = tf.matmul(dropout2, weight) + bias
        bn = batch_norm(linear, shape, is_training)
    return bn, is_training, keep_prob

def fully_connected_batchnorm(inputs, shape, is_training):
    weights = tf.get_variable("weight",
                              shape,
                              dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0,1))
    biases = tf.get_variable("bias", shape[-1])
    linear = tf.matmul(inputs, weights) + biases
    bn = batch_norm(linear, shape, is_training)
    return bn
    

def init_config():
    frame_buffer_noisy = np.array([], dtype=np.float32)

    config = { 
                'uid':0, 
                'frame_buffer_noisy':frame_buffer_noisy,
            } 
    return config


def fill_feed_dict(noisy_pl, config, noisy_file, shuffle):

    ark_dict_noisy, uid_new = read_mats(config['uid'], 0, noisy_file)
    if not ark_dict_noisy:
        return ({}, {}, [])
    id_noisy = list(ark_dict_noisy.keys())[0]
    mats_noisy = ark_dict_noisy[id_noisy]
    mats2_noisy = np.vstack(mats_noisy)
    

            
    frame_buffer_noisy = mats2_noisy
    frame_buffer_noisy = np.pad(frame_buffer_noisy,
                                    ((a.context,),(0,)),
                                    'constant',
                                    constant_values=0)
 
    config = {  'uid':uid_new,
                'frame_buffer_noisy':frame_buffer_noisy,
             } 
    noisy_batch = np.stack((frame_buffer_noisy[i:i+1+2*a.context,].flatten() for i in range(mats2_noisy.shape[0])), axis = 0)
    feed_dict = {noisy_pl:noisy_batch}
    return (feed_dict, config, id_noisy)
        

    
    
def placeholder_inputs():
    noisy_placeholder = tf.placeholder(tf.float32, shape=(None,a.input_featdim*(2*a.context+1)))
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    return noisy_placeholder


def run_generate():
    config = init_config() 
    totframes = 0

    with tf.Graph().as_default():
        noisy_pl = placeholder_inputs()
        predictions, is_training, keep_prob = create_generator(noisy_pl)
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, 'model/model.ckpt58431')

        start_time = time.time()
        step = 0
        while(True):
            feed_dict, config, id_noisy = fill_feed_dict(noisy_pl, config, a.noisy_train_file, shuffle=False)
            if not id_noisy:
                break
            feed_dict[keep_prob] = 1.0
            feed_dict[is_training] = False

            value = sess.run(predictions, feed_dict=feed_dict)
            kaldi_write_mats("reconstructed_feats.ark", bytes(id_noisy,'utf-8'), value)
    sess.close() 
    duration = time.time() - start_time

def main():
    run_generate()
    
if __name__=='__main__':
    main()    
    

