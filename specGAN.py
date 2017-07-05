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
from data_io import read_kaldi_ark_from_scp
from six.moves import xrange 

Model = collections.namedtuple("Model", "outputs, discrim_loss, gen_loss_GAN, gen_loss_objective, gen_loss, discrim_train, gd_train,misc")

data_base_dir = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument("--noisy_train_file", default="data-spectrogram/train_si84_noisy/feats.scp", help="The input feature file for training")
parser.add_argument("--noisy_dev_file", default="data-spectrogram/dev_dt_05_noisy/feats.scp", help="The input feature file for cross-validation")
parser.add_argument("--clean_train_file", default="data-spectrogram/train_si84_clean/feats.scp", help="The feature file for clean training labels")
parser.add_argument("--clean_dev_file", default="data-spectrogram/dev_dt_05_clean/feats.scp", help="The feature file for clean cross-validation labels")
parser.add_argument("--buffer_size", default=10, type=int)
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--exp_name", default=None, help="directory with checkpoint to resume training from or use for testing")
#Training
parser.add_argument("--lr", type=float, default = 0.0002, help = "initial learning rate")
parser.add_argument("--lr_decay", type=float, default=0.96, help = "learning rate decay")
parser.add_argument("--gan_weight", type=float, default=1.0, help = "weight of GAN loss in generator training")
parser.add_argument("--objective_weight", type=float, default=0.01, help = "weight of L1/MSE loss in generator training")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term")
#Model
parser.add_argument("--gen_objective", type=str, default="mse", choices=["mse","l1"])
parser.add_argument("--objective", type=str, default="mse", choices=["mse", "adv", "l1"])
parser.add_argument("--normalize_input", type=str, default="no", choices=["no", "sigmoid", "tanh"], help = "if no, do not normalize inputs; if sigmoid, normalize between [0,1], if tanh, normalize between [-1,1]")
parser.add_argument("--normalize_target", type=str, default="no", choices=["no", "sigmoid", "tanh"], help = "if no, do not normalize inputs; if sigmoid, normalize between [0,1], if tanh, normalize between [-1,1]")
parser.add_argument("--discrim_cond", type=str, default="full", choices=["full", "central"], help = "determines the form of the conditioning input to the discriminator; \"full\" uses full context window, and \"central\" uses only the central frame")
parser.add_argument("--nlayers", type=int, default=2)
parser.add_argument("--gen_units", type=int, default=2048)
parser.add_argument("--disc_units", type=int, default=1024)
parser.add_argument("--input_featdim", type=int, default=257)
parser.add_argument("--output_featdim", type=int, default=257)
parser.add_argument("--context", type=int, default=3)
parser.add_argument("--epsilon", type=float, default=1e-3, help="parameter for batch normalization")
parser.add_argument("--decay", type=float, default=0.999, help="parameter for batch normalization")
parser.add_argument("--num_steps_per_decay", type=int, default=5312*10, help="number of steps after learning rate decay")
parser.add_argument("--decay_rate", type=float, default=0.96, help="learning rate decay")
parser.add_argument("--max_global_norm", type=float, default=5.0, help="global max norm for clipping")
parser.add_argument("--keep_prob", type=float, default=0.4, help="keep percentage of neurons")
parser.add_argument("--no_test_dropout", action='store_true', default=False, help="if enabled, DO NOT use dropout during test")
parser.add_argument("--test_popnorm", action='store_true', default=False, help="if enabled, use population normalization instead of batch normalization at test")
#Generator Noise options 
parser.add_argument("--add_noise", action='store_true', default=False, help="if enabled, adds noise to the generator input")
parser.add_argument("--noise_dim", type=int, default=256, help="dimension of noise added")
parser.add_argument("--noise_mean", type=float, default=0, help="mean of the gaussian noise distribution")
parser.add_argument("--noise_std", type=float, default=1.0, help="std deviation of the gaussian noise distribution")

parser.add_argument("--patience", type=int, default=5312*10, help="patience interval to keep track of improvements")
parser.add_argument("--patience_increase", type=int, default=2, help="increase patience interval on improvement")
parser.add_argument("--improvement_threshold", type=float, default=0.995, help="keep track of validation error improvement")
a = parser.parse_args()
in_min = 0
in_max = 0
tgt_min = 0
tgt_max = 0

def read_mats(uid, offset, file_name):
    #Read a buffer containing buffer_size*batch_size+offset 
    #Returns a line number of the scp file
    scp_fn = path.join(data_base_dir, file_name)
    ark_dict,uid = read_kaldi_ark_from_scp(uid, offset, a.batch_size, a.buffer_size, scp_fn, data_base_dir)
    return ark_dict,uid

def mse_loss(predictions, labels):
  mse = tf.reduce_mean(tf.squared_difference(predictions, labels))
  return mse

def l1_loss(predictions, labels):
  loss = tf.reduce_mean(tf.abs(labels - predictions))
  return loss

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)



def batch_norm(x, shape, training):
    '''Assume 2d [batch, values] tensor'''

    scale = tf.get_variable('scale', shape[-1],  dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02)) # initializer=tf.constant_initializer(0.1))
    offset = tf.get_variable('offset', shape[-1],  initializer=tf.zeros_initializer())

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


def create_generator(generator_inputs, keep_prob, is_training):
    # Hidden 1
    with tf.variable_scope('hidden1'):
        if a.add_noise:
            shape = [(a.input_featdim*(2*a.context+1))+a.noise_dim, a.gen_units]
        else:
            shape = [a.input_featdim*(2*a.context+1), a.gen_units]
        weight = tf.get_variable("weight", shape, dtype=tf.float32, initializer = tf.random_normal_initializer(0,0.02))
        bias = tf.get_variable("bias", shape[-1], initializer=tf.constant_initializer(0.0))
        linear = tf.matmul(generator_inputs, weight) + bias
        bn = batch_norm(linear, shape, is_training)
        hidden = lrelu(bn,0.2)
        dropout1 = tf.nn.dropout(hidden, keep_prob)
    # Hidden 2
    with tf.variable_scope('hidden2'):
        shape = [a.gen_units, a.gen_units]
        weight = tf.get_variable("weight", shape, dtype=tf.float32, initializer = tf.random_normal_initializer(0,0.02))
        bias = tf.get_variable("bias", shape[-1], initializer=tf.constant_initializer(0.0))
        linear = tf.matmul(dropout1, weight) + bias
        bn = batch_norm(linear, shape, is_training)
        hidden = lrelu(bn,0.2)
        dropout2 = tf.nn.dropout(hidden, keep_prob)
    # Linear
    with tf.variable_scope('linear'):
        shape = [a.gen_units, a.output_featdim]
        weight = tf.get_variable("weight", shape, dtype=tf.float32, initializer = tf.random_normal_initializer(0,0.02))
        bias = tf.get_variable("bias", shape[-1], initializer=tf.constant_initializer(0.0))
        linear = tf.matmul(dropout2, weight) + bias
        out = batch_norm(linear, shape, is_training)
        if a.normalize_target == "sigmoid":
            out = tf.sigmoid(out)
        elif a.normalize_target == "tanh":
            out = tf.tanh(out)
    return out

def fully_connected_batchnorm(inputs, shape, is_training):
    weights = tf.get_variable("weight",
                              shape,
                              dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0,0.02))
    biases = tf.get_variable("bias", shape[-1], initializer=tf.constant_initializer(0.0))
    ab = tf.matmul(inputs, weights) + biases
    bn = batch_norm(ab, shape, is_training)
    return bn
    
def create_discriminator(discrim_inputs, discrim_targets, keep_prob, is_training):

    input = tf.concat([discrim_inputs, discrim_targets], axis = 1)
    with tf.variable_scope('discrim_layer1'):
        cond_input_frames = 0 # this should always get set correctly below
        if a.discrim_cond == "full":
            cond_input_frames = 2*a.context+1
        elif a.discrim_cond == "central":
            cond_input_frames = 1
        linear = fully_connected_batchnorm(input,
                                           (a.input_featdim*(cond_input_frames) + a.output_featdim,
                                            a.disc_units),
                                           is_training)
        relu = lrelu(linear,0.2)
        dropout = tf.nn.dropout(relu, keep_prob)
    with tf.variable_scope('discrim_layer2'):
        linear = fully_connected_batchnorm(dropout, (a.disc_units, a.disc_units), is_training)
        relu = lrelu(linear,0.2)
        dropout = tf.nn.dropout(relu, keep_prob)
    with tf.variable_scope('discrim_layer3'):
        linear = fully_connected_batchnorm(dropout, (a.disc_units, a.disc_units), is_training)
        relu = lrelu(linear,0.2)
        dropout = tf.nn.dropout(relu, keep_prob)
    with tf.variable_scope('discrim_layer4'):
        linear = fully_connected_batchnorm(dropout, (a.disc_units, a.disc_units), is_training)
        relu = lrelu(linear,0.2)
        dropout = tf.nn.dropout(relu, keep_prob)
    with tf.variable_scope('discrim_layer5'):
        linear = fully_connected_batchnorm(dropout, (a.disc_units, a.disc_units), is_training)
        relu = lrelu(linear,0.2)
        dropout = tf.nn.dropout(relu, keep_prob)
    with tf.variable_scope('discrim_layer6'):
        linear = fully_connected_batchnorm(dropout, (a.disc_units, 1), is_training)
        out = tf.sigmoid(linear)
    return out

def create_adversarial_model(inputs, targets, keep_prob, is_training):

    global_step = tf.contrib.framework.get_or_create_global_step()

    disc_cond_inputs = None
    if a.discrim_cond == "full":
        disc_cond_inputs = inputs
    elif a.discrim_cond == "central":
        disc_cond_inputs = tf.slice(inputs, [0, a.context*a.input_featdim], [-1, a.input_featdim])
        
    with tf.variable_scope('generator'):
        outputs = create_generator(inputs, keep_prob, is_training)

    with tf.name_scope('real_discriminator'):
        with tf.variable_scope('discriminator'):
            predict_real = create_discriminator(disc_cond_inputs, targets, keep_prob, is_training)

    with tf.name_scope('fake_discriminator'):
        with tf.variable_scope('discriminator', reuse = True):
            predict_fake = create_discriminator(disc_cond_inputs, outputs, keep_prob, is_training)

    # loss functions from pix2pix
    with tf.name_scope('discriminator_loss'):
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_real, labels=tf.ones_like(predict_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_fake, labels=tf.zeros_like(predict_fake)))
        discrim_loss = d_loss_real + d_loss_fake

    with tf.name_scope('generator_loss'):
        gen_loss_GAN = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_fake, labels=tf.ones_like(predict_fake))) 
        if (a.gen_objective == "l1"):
            gen_loss_objective  = tf.reduce_mean(tf.abs(targets - outputs))
        elif (a.gen_objective == "mse"):
            gen_loss_objective = tf.reduce_mean(tf.squared_difference(outputs, targets))
            
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_objective * a.objective_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        #print("Discrim_tvars length: ", len(discrim_tvars))
        #print([i.name for i in discrim_tvars])
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        gd_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        gd_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        gd_grads_and_vars = gd_optim.compute_gradients(gen_loss, var_list=gd_tvars)
        gd_train = gd_optim.apply_gradients(gd_grads_and_vars)
        
        #g_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        #g_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        #g_grads_and_vars = g_optim.compute_gradients(gen_loss, var_list=g_tvars)
        #g_train = g_optim.apply_gradients(g_grads_and_vars)
        #even = tf.equal(tf.mod(global_step, 2), 0)
        #gen_train = tf.cond(even, lambda: gd_train, lambda: g_train)
            

            
    #ema = tf.train.ExponentialMovingAverage(decay=0.99)
    #update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1, gen_loss])
    incr_global_step = tf.assign(global_step, global_step+1)

            
    return Model(
        discrim_loss=(discrim_loss),
        gen_loss_GAN=(gen_loss_GAN),
        gen_loss_objective=(gen_loss_objective),
        gen_loss=(gen_loss),
        outputs=outputs,
        discrim_train=discrim_train,
        gd_train=gd_train,
        misc=tf.group(incr_global_step),
    )

def training(loss):
    trainables = tf.trainable_variables()
    grads = tf.gradients(loss,trainables)
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=a.max_global_norm)
    grad_var_pairs = zip(grads, trainables)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    #learning_rate = tf.train.exponential_decay(
    #        a.lr, global_step, a.num_steps_per_decay,
    #        a.decay_rate, staircase=True)
    #tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(a.lr,a.beta1)
    train_op = optimizer.apply_gradients(grad_var_pairs, global_step=global_step)
    return train_op

def init_config():
    offset_frames_noisy = np.array([], dtype=np.float32).reshape(0,a.input_featdim)
    offset_frames_clean = np.array([], dtype=np.float32).reshape(0,a.output_featdim)
    frame_buffer_clean = np.array([], dtype=np.float32)
    frame_buffer_noisy = np.array([], dtype=np.float32)
    A = np.array([], dtype=np.int32)

    config = {'batch_index':0, 
                'uid':0, 
                'offset':0, 
                'offset_frames_noisy':offset_frames_noisy, 
                'offset_frames_clean':offset_frames_clean, 
                'frame_buffer_clean':frame_buffer_clean, 
                'frame_buffer_noisy':frame_buffer_noisy, 
                'perm':A}
    return config


def fill_feed_dict(noisy_pl, clean_pl, config, noisy_file, clean_file, shuffle):

    batch_index = config['batch_index']
    offset_frames_noisy = config['offset_frames_noisy']
    offset_frames_clean = config['offset_frames_clean']
    A = config['perm']

    def create_buffer(uid, offset):
        ark_dict_noisy,uid_new= read_mats(uid,offset,noisy_file)
        ark_dict_clean,uid_new = read_mats(uid,offset,clean_file)

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
            
        if mats2_noisy.shape[0]>=(a.batch_size*a.buffer_size):
            offset_frames_noisy = mats2_noisy[a.batch_size*a.buffer_size:]
            mats2_noisy = mats2_noisy[:a.batch_size*a.buffer_size]
            offset_frames_clean = mats2_clean[a.batch_size*a.buffer_size:]
            mats2_clean = mats2_clean[:a.batch_size*a.buffer_size]
            offset = offset_frames_noisy.shape[0]
        return mats2_noisy, mats2_clean, uid_new, offset

    if batch_index==0:
        frame_buffer_noisy, frame_buffer_clean, uid_new, offset = create_buffer(config['uid'],
                                                                                config['offset'])
        if a.normalize_input == "sigmoid":
            frame_buffer_noisy = np.interp(frame_buffer_noisy, [in_min, in_max], [0.0, 1.0])
        elif a.normalize_input == "tanh":
            frame_buffer_noisy = np.interp(frame_buffer_noisy, [in_min, in_max], [-1.0, 1.0])

        if a.normalize_target == "sigmoid":
            frame_buffer_clean = np.interp(frame_buffer_clean, [tgt_min, tgt_max], [0.0, 1.0])
        elif a.normalize_target == "tanh":
            frame_buffer_clean = np.interp(frame_buffer_clean, [tgt_min, tgt_max], [-1.0, 1.0])

        frame_buffer_noisy = np.pad(frame_buffer_noisy,
                                    ((a.context,),(0,)),
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


    start = batch_index*a.batch_size
    end = min((batch_index+1)*a.batch_size,frame_buffer_clean.shape[0])
    config = {'batch_index':(batch_index+1)%a.buffer_size, 
                'uid':uid_new,
                'offset':offset, 
                'offset_frames_noisy':offset_frames_noisy,
                'offset_frames_clean':offset_frames_clean, 
                'frame_buffer_noisy':frame_buffer_noisy,
                'frame_buffer_clean':frame_buffer_clean, 
                'perm':A}
    noisy_batch = np.stack((frame_buffer_noisy[A[i]:A[i]+1+2*a.context,].flatten()
                            for i in range(start, end)), axis = 0)
    if a.add_noise:
        batch_z = np.random.normal(a.noise_mean, a.noise_std, [noisy_batch.shape[0], a.noise_dim]).astype(np.float32)
        temp = np.concatenate((noisy_batch, batch_z), axis=1)
        noisy_batch = temp
    feed_dict = {noisy_pl:noisy_batch, clean_pl:frame_buffer_clean[start:end]}
    return (feed_dict, config)
        

    
    
def placeholder_inputs():
    if a.add_noise:
        shape = (a.input_featdim*(2*a.context+1))+a.noise_dim
    else:
        shape = a.input_featdim*(2*a.context+1)
    noisy_placeholder = tf.placeholder(tf.float32, shape=(None,shape), name="noisy_placeholder")
    clean_placeholder = tf.placeholder(tf.float32, shape=(None,a.output_featdim), name="clean_placeholder")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    is_training = tf.placeholder(tf.bool, name="is_training")
    return noisy_placeholder, clean_placeholder, keep_prob, is_training

def do_eval(sess, gen_fetches, noisy_pl, clean_pl, is_training, keep_prob):
    config = init_config()
    tot_loss_epoch = 0
    tot_objective_loss_epoch = 0
    tot_gan_loss_epoch = 0
    totframes = 0

    start_time = time.time()
    while(True):
        feed_dict, config = fill_feed_dict(noisy_pl, clean_pl, config, a.noisy_dev_file, a.clean_dev_file, shuffle=False)
        
        feed_dict[is_training] = False if a.test_popnorm else True
        feed_dict[keep_prob] = 1.0 if a.no_test_dropout else a.keep_prob
        if feed_dict[noisy_pl].shape[0]<a.batch_size:
            if a.objective == "adv":
                result = sess.run(gen_fetches, feed_dict=feed_dict)
                tot_objective_loss_epoch += feed_dict[noisy_pl].shape[0]*result["objective_loss"]
                tot_gan_loss_epoch += feed_dict[noisy_pl].shape[0]*result["GAN_loss"]
            elif a.objective == "mse" or a.objective == "l1":
                result = sess.run(gen_fetches, feed_dict=feed_dict)
            tot_loss_epoch += feed_dict[noisy_pl].shape[0]*result["loss"]
            totframes += feed_dict[noisy_pl].shape[0]
            break
    
        if a.objective == "adv":
            result = sess.run(gen_fetches, feed_dict=feed_dict)
            tot_objective_loss_epoch += feed_dict[noisy_pl].shape[0]*result["objective_loss"]
            tot_gan_loss_epoch += feed_dict[noisy_pl].shape[0]*result["GAN_loss"]
        elif a.objective == "mse" or a.objective == "l1":
            result = sess.run(gen_fetches, feed_dict=feed_dict)
        tot_loss_epoch += feed_dict[noisy_pl].shape[0]*result["loss"]
        totframes += feed_dict[noisy_pl].shape[0]

    if (a.objective == "adv"):
        eval_objective_loss = float(tot_objective_loss_epoch)/totframes
        eval_gan_loss = float(tot_gan_loss_epoch)/totframes
    eval_correct = float(tot_loss_epoch)/totframes 
    duration = time.time() - start_time
    if (a.objective == "adv"):
        print ('loss = %.6f Objective_loss = %.6f GAN_loss = %.6f (%.3f sec)' % (eval_correct, eval_objective_loss, eval_gan_loss, duration))
    elif a.objective == "mse" or a.objective == "l1":
        print ('loss = %.6f (%.3f sec)' % (eval_correct, duration))
    return eval_correct, duration



def run_training():
    config = init_config() 
    if not os.path.isdir(a.exp_name):
        os.makedirs(a.exp_name)
    tot_loss_epoch = 0
    tot_objective_loss_epoch = 0
    tot_gan_loss_epoch = 0
    tot_discrim_loss_epoch = 0
    avg_loss_epoch = 0
    totframes = 0
    best_validation_loss = np.inf
    patience = a.patience
    with tf.Graph().as_default():
        noisy_pl, clean_pl, keep_prob, is_training = placeholder_inputs()
        disc_fetches = {}
        gen_fetches = {}
        if a.objective == "mse" or a.objective == "l1":
            with tf.variable_scope('generator'):
                predictions = create_generator(noisy_pl, keep_prob, is_training)
            if a.objective == "mse" :
                gen_fetches['loss'] = mse_loss(predictions, clean_pl)
            elif a.objective == "l1":
                gen_fetches['loss'] = l1_loss(predictions, clean_pl)
            gen_fetches['train'] = training(gen_fetches['loss'])
            # loss_val = fetches['loss']
            # train_op = training(loss_val)
        elif a.objective == "adv":
            model = create_adversarial_model(noisy_pl, clean_pl, keep_prob, is_training)
            disc_fetches['objective_loss'] = model.gen_loss_objective
            disc_fetches['GAN_loss'] = model.gen_loss_GAN
            disc_fetches['discrim_loss'] = model.discrim_loss
            disc_fetches['loss'] = model.gen_loss
            disc_fetches['discrim_train'] = model.discrim_train
            disc_fetches['misc'] = model.misc

            gen_fetches['objective_loss'] = model.gen_loss_objective
            gen_fetches['GAN_loss'] = model.gen_loss_GAN
            gen_fetches['discrim_loss'] = model.discrim_loss
            gen_fetches['loss'] = model.gen_loss
            gen_fetches['generator_train'] = model.gd_train
            gen_fetches['misc'] = model.misc

            #fetches['predict_real'] = model.predict_real
            #fetches['predict_fake'] = model.predict_fake
            # loss_val = fetches['loss']
            # tf.summary.scalar('loss', loss_val)
        #summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        all_vars = tf.global_variables()
        saver = tf.train.Saver([var for var in all_vars if 'generator' in var.name])
        sess = tf.Session()
        #summary_writer = tf.summary.FileWriter("log", sess.graph)

        sess.run(init)
        start_time = time.time()
        step = 0
        if a.normalize_input != "no":
            print("finding range of input values...")
            global in_min, in_max
            in_min, in_max = find_min_max(a.noisy_train_file)
        if a.normalize_target != "no":
            print("finding range of target values...")
            global tgt_min, tgt_max
            tgt_min, tgt_max = find_min_max(a.clean_train_file)
        
        while(True):
            feed_dict, config = fill_feed_dict(noisy_pl, clean_pl, config, a.noisy_train_file, a.clean_train_file, shuffle=True)
            feed_dict[keep_prob] = a.keep_prob
            feed_dict[is_training] = True

            if feed_dict[noisy_pl].shape[0]<a.batch_size:
                config = init_config()
            if a.objective == "adv": 
                result = sess.run(disc_fetches, feed_dict=feed_dict)
                result = sess.run(gen_fetches, feed_dict=feed_dict)
                result = sess.run(gen_fetches, feed_dict=feed_dict)
            elif a.objective == "mse" or a.objective == "l1":
                result = sess.run(gen_fetches, feed_dict=feed_dict)
 
            #result = sess.run(fetches, feed_dict=feed_dict)
            #if a.objective == "adv":
            #    print ('L1: %.2f  GAN: %.6f  Discrim: %.6f Loss:%.6f'
            #           % (result['L1_loss'], result['GAN_loss'], result['discrim_loss'], result['loss']))
            #    #print ('predict_real: ', result['predict_real'])
            #    #print ('predict_fake: ', result['predict_fake'])
            
            tot_loss_epoch += feed_dict[noisy_pl].shape[0]*result['loss']
            if (a.objective == "adv"): 
                tot_objective_loss_epoch += feed_dict[noisy_pl].shape[0]*result['objective_loss']
                tot_gan_loss_epoch += feed_dict[noisy_pl].shape[0]*result['GAN_loss']
                tot_discrim_loss_epoch += feed_dict[noisy_pl].shape[0]*result['discrim_loss']

            totframes += feed_dict[noisy_pl].shape[0]

            if (step+1)%5312 == 0:
                avg_loss_epoch = float(tot_loss_epoch)/totframes
                if (a.objective == "adv"):
                    avg_objective_loss_epoch = float(tot_objective_loss_epoch)/totframes
                    avg_gan_loss_epoch = float(tot_gan_loss_epoch)/totframes
                    avg_discrim_loss_epoch = float(tot_discrim_loss_epoch)/totframes
                tot_loss_epoch = 0   
                tot_objective_loss_epoch = 0
                tot_gan_loss_epoch = 0
                tot_discrim_loss_epoch = 0
                duration = time.time() - start_time
                start_time = time.time()
                print ('Step %d: loss = %.6f (%.3f sec)' % (step, avg_loss_epoch, duration))
                if a.objective == "adv":
                    print ('Objective: %.2f  GAN: %.6f  Discrim: %.6f Loss:%.6f'
                           % (avg_objective_loss_epoch, avg_gan_loss_epoch, avg_discrim_loss_epoch, avg_loss_epoch))

                    #print ('predict_real: ', result['predict_real'])
                    #print ('predict_fake: ', result['predict_fake'])
                #summary_str = sess.run(summary, feed_dict=feed_dict)
                #summary_writer.add_summary(summary_str, step)
                #summary_writer.flush()
                print ('Eval step:')
                eval_loss, duration = do_eval(sess, gen_fetches, noisy_pl,
                                              clean_pl, is_training, keep_prob)
                
                if eval_loss<best_validation_loss:
                    if eval_loss<best_validation_loss * a.improvement_threshold:
                        patience = max(patience, (step+1)* a.patience_increase)
                    best_validation_loss = eval_loss
                    best_iter = step
                save_path = saver.save(sess, os.path.join(a.exp_name,"model.ckpt"), global_step=step)
            if patience<=step:
                break
            step = step + 1

def find_min_max(scp_file):
    minimum = float("inf")
    maximum = -float("inf")
    uid = 0
    offset = 0
    ark_dict, uid = read_mats(uid, offset, scp_file)
    while ark_dict:
        for key in ark_dict.keys():
            mat_max = np.amax(ark_dict[key])
            mat_min = np.amin(ark_dict[key])
            if mat_max > maximum:
                maximum = mat_max
            if mat_min < minimum:
                minimum = mat_min
        ark_dict, uid = read_mats(uid, offset, scp_file)
    print("min:", minimum, "max:", maximum)
    return minimum, maximum

def main():
    run_training()
    
if __name__=='__main__':
    main()    
    

