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

data_base_dir = "/data/data2/scratch/bagchid/specGAN-tf"
def read_mats(file_name):
	#Read a buffer containing 10*batch_size 
	#Returns a utt_id and frame number
	scp_fn = path.join(data_base_dir, file_name)
	ark_dict = read_kaldi_ark_from_scp(scp_fn, data_base_dir)
	return ark_dict

def create_batch(ark_dict, batch_number, batch_size):
	def create_permutations(ids):
		return np.random.permutation(ids)
    	
	#idnums = [i for (i,j) in enumerate(ids)]

	rperm = create_permutations(idnums)
	sub_ids = [ids[rperm[i]] for i in xrange(batch_number*batch_size, (batch_number+1)*batch_size)]
	mats = [ark_dict[i].T for i in sub_ids]
	return mats	

def create_model():
	def create_generator(generator_inputs):
		layers = []
		with tf.variable_scope("gen_layer_1"):
			output = dense(generator_inputs)
			layers.append(output)
    	for _ in range(nlayers-1):
			with tf.variable_scope("gen_layer_%d" % (len(layers) + 1)):
				rectified = dense(layers[-1])
				output = batchnorm(convolved)
				layers.append(output)
		return layers[-1] 

	def create_discriminator(discrim_inputs, discrim_targets):
		n_layers = 6
		layers = []
		with tf.variable_scope("disc_layer_1"):
			output = dense(generator_inputs)
			layers.append(output)
    	for _ in range(nlayers-1):
			with tf.variable_scope("disc_layer_%d" % (len(layers) + 1)):
				rectified = dense(layers[-1])
				output = batchnorm(convolved)
				layers.append(output)
		return layers[-1]

	with tf.variable_scope("generator") as scope:
		out_channels = int(targets.get_shape()[-1])
		outputs = create_generator(inputs, out_channels)

	with tf.name_scope("discriminator_real"):
		with tf.variable_scope("discriminator"):
			predict_real = create_discriminator(inputs, targets)

	with tf.name_scope("discriminator_fake"):
		with tf.variable_scope("discriminator", reuse=True):
			predict_fake = create_discriminator(inputs, outputs)
	with tf.name_scope("discriminator_loss"):
		discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

	with tf.name_scope("generator_loss"):
		gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
		gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
		gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

def main():
	ark_dict_noisy = read_mats("data-spectrogram/train_si84_noisy/feats.scp")
	ark_dict_clean = read_mats("data-spectrogram/train_si84_clean/feats.scp")

	ids_noisy = sorted(ark_dict_noisy.keys())
	mats_noisy = [ark_dict_noisy[i] for i in ids_noisy]
	mats2_noisy = np.vstack(mats_noisy)

	ids_clean = sorted(ark_dict_clean.keys())
	mats_clean = [ark_dict_clean[i] for i in ids_clean]
	mats2_clean = np.vstack(mats_clean)


	nframes = mats2_noisy.shape[0]
	sizes = [ark_dict_noisy[i].shape[0] for i in ids_noisy]
	print(nframes)
	batch_size=1024*1024
	nepochs=1	
	for epoch in range(nepochs):
		for batch_number in range(int(nframes/batch_size) + 1):
			end_frame = min((batch_number+1)*batch_size-1, nframes-1)
			tf_mats2_noisy = tf.convert_to_tensor(mats2_noisy[batch_number*batch_size:end_frame], dtype=tf.float32)
			tf_mats2_clean = tf.convert_to_tensor(mats2_clean[batch_number*batch_size:end_frame], dtype=tf.float32)
	ark_dict.clear()


if __name__=='__main__':
	main()	
	

