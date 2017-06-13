from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

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
from data_io import read_kaldi_ark_from_scp
from six.moves import xrange 

data_base_dir = "/data/data2/scratch/bagchid/specGAN"
def read_mats(uid, offset, batch_size, file_name):
    #Read a buffer containing 10*batch_size+offset 
    #Returns a line number of the scp file
    scp_fn = path.join(data_base_dir, file_name)
    ark_dict,uid = read_kaldi_ark_from_scp(uid, offset, batch_size, scp_fn, data_base_dir)
    return ark_dict,uid

def loss(predictions, labels):
  mse = tf.reduce_mean(tf.squared_difference(predictions, labels))
  return mse


def batch_norm(x, name_scope, shape, training, epsilon=1e-3, decay=0.999):
    '''Assume 2d [batch, values] tensor'''

    with tf.variable_scope(name_scope):
        scale = tf.get_variable('scale', shape[-1], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', shape[-1])

        pop_mean = tf.get_variable('pop_mean', shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
        pop_var = tf.get_variable('pop_var', shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
        batch_mean, batch_var = tf.nn.moments(x, [0])

        train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(training, batch_statistics, population_statistics)


def create_generator(generator_inputs):

    #Create additional placeholder inputs
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    # Hidden 1
    with tf.variable_scope('hidden1'):
        shape = [257, 1024]
        weight = tf.get_variable("weight", shape, dtype=tf.float32, initializer = tf.random_normal_initializer(0,1))
        bias = tf.get_variable("bias", shape[-1])
        linear = tf.matmul(generator_inputs, weight) + bias
        bn = batch_norm(linear, 'hidden1', shape, is_training)
        hidden = tf.nn.relu(bn)
        dropout1 = tf.nn.dropout(hidden, keep_prob)
    # Hidden 2
    with tf.variable_scope('hidden2'):
        shape = [1024, 1024]
        weight = tf.get_variable("weight", shape, dtype=tf.float32, initializer = tf.random_normal_initializer(0,1))
        bias = tf.get_variable("bias", shape[-1])
        linear = tf.matmul(dropout1, weight) + bias
        bn = batch_norm(linear, 'hidden2', shape, is_training)
        hidden = tf.nn.relu(bn)
        dropout2 = tf.nn.dropout(hidden, keep_prob)
    # Linear
    with tf.variable_scope('linear'):
        shape = [1024, 257]
        weight = tf.get_variable("weight", shape, dtype=tf.float32, initializer = tf.random_normal_initializer(0,1))
        bias = tf.get_variable("bias", shape[-1])
        linear = tf.matmul(dropout2, weight) + bias
        bn = batch_norm(linear, 'linear', shape, is_training)
    return bn, is_training, keep_prob
    

def training(loss, initial_learning_rate, num_steps_per_decay, decay_rate, max_global_norm=5.0):
    trainables = tf.trainable_variables()
    grads = tf.gradients(loss,trainables)
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
    grad_var_pairs = zip(grads, trainables)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(
            initial_learning_rate, global_step, num_steps_per_decay,
            decay_rate, staircase=True)
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grad_var_pairs, global_step=global_step)
    return train_op

# do we need the inputs from noisy_pl and clean_pl?        
def fill_feed_dict(noisy_pl, clean_pl, config, noisy_file, clean_file, shuffle):

    batch_index = config['batch_index']
    batch_size = config['batch_size']
    offset_frames_noisy = config['offset_frames_noisy']
    offset_frames_clean = config['offset_frames_clean']
    lr_ctx = config['lr_ctx']
    A = config['perm']

    def create_buffer(uid, offset):
        ark_dict_noisy,uid_new= read_mats(uid,offset,batch_size,noisy_file)
        ark_dict_clean,uid_new = read_mats(uid,offset,batch_size,clean_file)

        ids_noisy = sorted(ark_dict_noisy.keys())
        mats_noisy = [ark_dict_noisy[i] for i in ids_noisy]
        mats2_noisy = np.vstack(mats_noisy)
        nonlocal offset_frames_noisy
        mats2_noisy = np.concatenate((offset_frames_noisy,mats2_noisy),axis=0)

        ids_clean = sorted(ark_dict_clean.keys())
        mats_clean = [ark_dict_clean[i] for i in ids_clean]
        mats2_clean = np.vstack(mats_clean)
        nonlocal offset_frames_clean
        mats2_clean = np.concatenate((offset_frames_clean,mats2_clean),axis=0)
            
        if mats2_noisy.shape[0]>=(batch_size*10):
            offset_frames_noisy = mats2_noisy[batch_size*10:]
            mats2_noisy = mats2_noisy[:batch_size*10]
            offset_frames_clean = mats2_clean[batch_size*10:]
            mats2_clean = mats2_clean[:batch_size*10]
            offset = offset_frames_noisy.shape[0]
        return mats2_noisy, mats2_clean, uid_new, offset

    if batch_index==0:
        frame_buffer_noisy, frame_buffer_clean, uid_new, offset = create_buffer(config['uid'],
                                                                                config['offset'])
        frame_buffer_noisy = np.pad(frame_buffer_noisy,
                                    ((lr_ctx,),(0,)),
                                    'constant',
                                    constant_values=0)
        if shuffle==True:
            A = np.random.permutation(frame_buffer_clean.shape[0])
        else:
            A = np.arange(frame_buffer_clean.shape[0])
        # we don't permute the noisy frames because we need to preserve context;
        # we take matching windows in the assignment to noisy_batch below;
        # this means we pass the permutation in config
        frame_buffer_clean = frame_buffer_clean[A]
 
    else:
        frame_buffer_noisy = config['frame_buffer_noisy']
        frame_buffer_clean = config['frame_buffer_clean']
        uid_new = config['uid']
        offset = config['offset']

    start = batch_index*batch_size
    end = min((batch_index+1)*batch_size,frame_buffer_noisy.shape[0])
    config = {'batch_size':batch_size, 'batch_index':(batch_index+1)%10, 'uid':uid_new,
              'offset':offset, 'offset_frames_noisy':offset_frames_noisy,
              'offset_frames_clean':offset_frames_clean, 'frame_buffer_noisy':frame_buffer_noisy,
              'frame_buffer_clean':frame_buffer_clean, 'lr_ctx':lr_ctx, 'perm':A}
    noisy_batch = np.stack(frame_buffer_noisy[A[i]:A[i]+1+2*lr_ctx] for i in range(start, end),
                           axis = 0)
    feed_dict = {noisy_pl:noisy_batch, clean_pl:frame_buffer_clean[start:end]}
    return (feed_dict, config)
        

    
    
def placeholder_inputs(num_feats, lr_ctx):
    noisy_placeholder = tf.placeholder(tf.float32, shape=(None,num_feats,2*lr_ctx+1))
    clean_placeholder = tf.placeholder(tf.float32, shape=(None,num_feats))
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    return noisy_placeholder, clean_placeholder

def do_eval(sess, loss_val, noisy_pl, clean_pl, is_training, keep_prob):
    batch_size=1024
    batch_index = 0
    offset_frames_noisy = np.array([], dtype=np.float32).reshape(0,257)
    offset_frames_clean = np.array([], dtype=np.float32).reshape(0,257)
    frame_buffer_clean = np.array([], dtype=np.float32)
    frame_buffer_noisy = np.array([], dtype=np.float32)
    tot_loss_epoch = 0 
    totframes = 0
    start_time = time.time()
    A = np.array([], dtype=np.int32)
    
    config = {'batch_size':batch_size, 'batch_index':0, 'uid':0, 'offset':0, 'offset_frames_noisy':offset_frames_noisy, 'offset_frames_clean':offset_frames_clean, 'frame_buffer_clean':frame_buffer_clean, 'frame_buffer_noisy':frame_buffer_noisy, 'lr_ctx':5, 'perm':A}

    noisy_pl, clean_pl = placeholder_inputs(257, config['lr_ctx'])
    while(True):
        feed_dict, config = fill_feed_dict(noisy_pl, clean_pl, config, "data-spectrogram/dev_dt_05_noisy/feats.scp", "data-spectrogram/dev_dt_05_clean/feats.scp", shuffle=False)
        feed_dict[is_training] = False
        feed_dict[keep_prob] = 1.0
        if feed_dict[noisy_pl].shape[0]<batch_size:
            loss_value = sess.run(loss_val, feed_dict=feed_dict)
            tot_loss_epoch += feed_dict[noisy_pl].shape[0]*loss_value
            totframes += feed_dict[noisy_pl].shape[0]
            break

        loss_value = sess.run(loss_val, feed_dict=feed_dict)
        tot_loss_epoch += feed_dict[noisy_pl].shape[0]*loss_value
        totframes += feed_dict[noisy_pl].shape[0]

    eval_correct = float(tot_loss_epoch)/totframes
    duration = time.time() - start_time
    print ('loss = %.2f (%.3f sec)' % (eval_correct, duration))
    return eval_correct, duration



def run_training():
    batch_size=1024
    batch_index = 0
    offset_frames_noisy = np.array([], dtype=np.float32).reshape(0,257)
    offset_frames_clean = np.array([], dtype=np.float32).reshape(0,257)
    frame_buffer_clean = np.array([], dtype=np.float32)
    frame_buffer_noisy = np.array([], dtype=np.float32)
    A = np.array([], dtype=np.int32)
    phase = True
    # Is there a reason we need to initialize this here when it gets initialized below?
    config = {'batch_size':batch_size, 'batch_index':0, 'uid':0, 'offset':0, 'offset_frames_noisy':offset_frames_noisy, 'offset_frames_clean':offset_frames_clean, 'frame_buffer_clean':frame_buffer_clean, 'frame_buffer_noisy':frame_buffer_noisy, 'lr_ctx':5, 'perm':A}

    os.makedirs("model")
    noisy_pl, clean_pl = placeholder_inputs(257, config['lr_ctx'])
    tot_loss_epoch = 0
    avg_loss_epoch = 0
    totframes = 0
    keep_prob = 0.4
    best_validation_loss = np.inf
    improvement_threshold = 0.995
    patience = 10*5312
    patience_increase = 2

    with tf.Graph().as_default():
        noisy_pl, clean_pl = placeholder_inputs(257, config['lr_ctx'])
        predictions, is_training, keep_prob = create_generator(noisy_pl)
        loss_val = loss(predictions, clean_pl)
        train_op = training(loss_val,0.08,5312*10,0.96)
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter("log", sess.graph)

        sess.run(init)
        start_time = time.time()
        step = 0
        while(True):
            feed_dict, config = fill_feed_dict(noisy_pl, clean_pl, config, "data-spectrogram/train_si84_noisy/feats.scp", "data-spectrogram/train_si84_clean/feats.scp", shuffle=True)
            feed_dict[keep_prob] = 0.4
            feed_dict[is_training] = True

            if feed_dict[noisy_pl].shape[0]<batch_size:
                batch_index = 0
                offset_frames_noisy = np.array([], dtype=np.float32).reshape(0,257)
                offset_frames_clean = np.array([], dtype=np.float32).reshape(0,257)
                frame_buffer_clean = np.array([], dtype=np.float32)
                frame_buffer_noisy = np.array([], dtype=np.float32)
                A = np.array([], dtype=np.int32)
                config = {'batch_size':batch_size, 'batch_index':0, 'uid':0, 'offset':0, 'offset_frames_noisy':offset_frames_noisy, 'offset_frames_clean':offset_frames_clean, 'frame_buffer_clean':frame_buffer_clean, 'frame_buffer_noisy':frame_buffer_noisy, 'lr_ctx':5, 'perm':A}



            _, loss_value = sess.run([train_op, loss_val], feed_dict=feed_dict)
            tot_loss_epoch += feed_dict[noisy_pl].shape[0]*loss_value
            totframes += feed_dict[noisy_pl].shape[0]

            if (step+1)%5312 == 0:
                avg_loss_epoch = float(tot_loss_epoch)/totframes
                tot_loss_epoch = 0   
                duration = time.time() - start_time
                start_time = time.time()
                print ('Step %d: loss = %.2f (%.3f sec)' % (step, avg_loss_epoch, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
                print ('Eval step:')
                eval_loss, duration = do_eval(sess, loss_val, noisy_pl, clean_pl, is_training, keep_prob)
                
                if eval_loss<best_validation_loss:
                    if eval_loss<best_validation_loss * improvement_threshold:
                        patience = max(patience, (step+1)* patience_increase)
                    best_validation_loss = eval_loss
                    best_iter = step
                    save_path = saver.save(sess, "model/model.ckpt"+str(step))
            if patience<=step:
                break
            step = step + 1



def main():
    run_training()
    
if __name__=='__main__':
    main()    
    

