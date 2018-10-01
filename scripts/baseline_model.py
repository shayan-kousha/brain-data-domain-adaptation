#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 17:01:57 2018

@author: shayan
"""
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import cohen_kappa_score
from mne import Epochs, pick_types, find_events
from mne.io import concatenate_raws, read_raw_edf, Raw, find_edf_events
from scipy.io import matlab
import matplotlib.pyplot as plt

def load_data(file_path):
    tmin, tmax = -1., 4.
    event_id = dict(left_hand=769, right_hand=770, both_feet=771, tongue=772)
    raw = read_raw_edf(file_path, eog= [22, 23, 24], stim_channel="auto")

    events = find_edf_events(raw)
    events = np.stack((events[1], events[3], events[2]), 1).astype(np.int64)
    
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs_train.events[:, -1] - 769
    
    indices = np.arange(epochs_train.get_data().shape[0])   
    np.random.shuffle(indices)
    
    data = epochs_train.get_data()[:, :, :, np.newaxis]
    
    return data[indices], labels[indices]

def brain_model(features, num_classes=4):
    conv1 = tf.layers.conv2d(inputs=features, filters = 20, kernel_size=(1, 5), strides=1, activation=tf.nn.relu)
    batch1 = tf.layers.batch_normalization(conv1)
        
    conv2 = tf.layers.conv2d(inputs=batch1, filters = 20, kernel_size=(22, 1), strides=1, activation=tf.nn.relu)
    batch2 = tf.layers.batch_normalization(conv2)
    pool2 = tf.layers.max_pooling2d(inputs=batch2, pool_size=(1, 3), strides=2)
    
    conv3 = tf.layers.conv2d(inputs=pool2, filters = 50, kernel_size=(1, 10), strides=1, activation=tf.nn.relu)
    batch3 = tf.layers.batch_normalization(conv3)
    pool3 = tf.layers.max_pooling2d(inputs=batch3, pool_size=(1, 3), strides=2)
    
    
    conv4 = tf.layers.conv2d(inputs=pool3, filters = 100, kernel_size=(1, 10), strides=1, activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=(1, 3), strides=2)
    
    conv5 = tf.layers.conv2d(inputs=pool4, filters = 200, kernel_size=(1, 10), strides=1, activation=tf.nn.relu)
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=(1, 3), strides=2)
      
    conv6 = tf.layers.conv2d(inputs=pool5, filters = 4, kernel_size=(1, 6), strides=1, activation=None)
    
    logits = tf.reshape(conv6, [-1, 4])
    
    return logits

def train_loop(file_path):
    data, labels = load_data(file_path)
    data *= 10**6
    
    epoch_number = 10
    batch_size = 96
    num_batches = 3
    
    d = tf.placeholder(tf.float32, [batch_size, 22, 251, 1])
    l = tf.placeholder(tf.int32, [batch_size])

    model = brain_model(d, 4)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=l, logits=model)
    
    
    lr = tf.placeholder(tf.float32)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr)
    
    
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
         
        saver = tf.train.Saver()
        
        try:
            saver.restore(sess, "../checkpoints/model_epoch_9.ckpt")
        except Exception as e:
            print (e)
            
        print("{} Start training...".format(datetime.now()))

        for epoch in range(0, epoch_number):
            all_loss = []
            
            for step in range(num_batches):
                train_data = data[step*batch_size: (step+1)*batch_size]
                train_label = labels[step*batch_size: (step+1)*batch_size]
                
                logits, _, ret_loss = sess.run([model, train_op, loss], feed_dict={d: train_data, lr:0.08, l:train_label})
                all_loss.append(ret_loss)
                
            print("Epoch {}, Loss {:.6f}, Kappa {}".format(epoch, np.mean(all_loss), cohen_kappa_score(np.argmax(logits, 1), train_label)))
            
        saver.save(sess, "../checkpoints/model_epoch_" + str(epoch) + ".ckpt" )
        
def main(argv):
    file_path = "../data/BCICIV_2a_gdf/A09T.gdf"
    train_loop(file_path)
    
if __name__ == "__main__":
    tf.app.run()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    