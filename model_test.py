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
parser.add_argument("--noisy_file", default="data-spectrogram/dev_dt_05_noisy/feats.scp", help="The input feature file for training")
parser.add_argument("--meta_file", default="model/model.ckpt-5311.meta", help = "The meta file to load from")
parser.add_argument("--checkpoint", default="model/model.ckpt-5311")
parser.add_argument("--out_file", default="reconstructed_feats.ark", help = "The file to write the features to")
parser.add_argument("--context", type=int, default=5)

parser.add_argument("--buffer_size", default=1, type=int)
parser.add_argument("--batch_size", default=1, type=int)
a = parser.parse_args()


def read_mats(uid, offset, file_name):
    #Read a buffer containing buffer_size*batch_size+offset 
    #Returns a line number of the scp file
    scp_fn = path.join(data_base_dir, file_name)
    ark_dict,uid = read_kaldi_ark_from_scp(uid, offset, a.batch_size, a.buffer_size, scp_fn, data_base_dir)
    return ark_dict,uid

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
        

    
    
def run_generate():
    config = init_config() 
    sess = tf.Session()
    saved_model = tf.train.import_meta_graph(a.meta_file)
    saved_model.restore(sess,a.checkpoint)
    
    graph = tf.get_default_graph()
    noisy_pl = graph.get_tensor_by_name("noisy_placeholder:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    is_training = graph.get_tensor_by_name("is_training:0")
    

    predictions = graph.get_tensor_by_name("linear/cond/Merge:0")

    start_time = time.time()
    step = 0
    while(True):
        feed_dict, config, id_noisy = fill_feed_dict(noisy_pl, config, a.noisy_file, shuffle=False)
        if not id_noisy:
            break
        feed_dict[keep_prob] = 1.0
        feed_dict[is_training] = False

        value = sess.run(predictions, feed_dict=feed_dict)
        kaldi_write_mats(a.out_file, bytes(id_noisy,'utf-8'), value)
    sess.close() 
    duration = time.time() - start_time

def main():
    run_generate()
    
if __name__=='__main__':
    main()    
    

