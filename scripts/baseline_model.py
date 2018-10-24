#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 17:01:57 2018

@author: shayan
"""
import helper_functions as hf
from load_data import BCICompetition4Set2A
from collections import OrderedDict
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import cohen_kappa_score
from mne import Epochs, pick_types
from mne.io import read_raw_edf, find_edf_events
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    event_id = dict(left_hand=769, right_hand=770, both_feet=771, tongue=772)
    
    tmin, tmax = -0.5, 4
    
    raw = read_raw_edf(file_path, eog= [22, 23, 24], stim_channel="auto", preload=True)
    raw.filter(0.5, 38.)
    events = find_edf_events(raw)
    events = np.stack((events[1], events[3], events[2]), 1).astype(np.int64)
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
    epochs_train = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    labels = epochs_train.events[:, -1] - 769
    
    indices = np.arange(labels.shape[0])
    np.random.shuffle(indices)
    
    return epochs_train.get_data()[:, :, :, np.newaxis][indices], labels[indices]

def BCI_load_data(filename, labels_filename):
    ival = [-500, 4000]
    
    train_loader = BCICompetition4Set2A(filename=filename, labels_filename=labels_filename)
    train_cnt = train_loader.load()
    train_cnt = train_cnt.drop_channels(['STI 014', 'EOG-left', 'EOG-central', 'EOG-right'])
    assert len(train_cnt.ch_names) == 22
    
    train_cnt = hf.mne_apply(lambda a: a * 1e6, train_cnt)
    train_cnt = hf.mne_apply(lambda a: hf.bandpass_cnt(a, 0, 38.0, train_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), train_cnt)
    train_cnt = hf.mne_apply(lambda a: hf.exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T, train_cnt)
    
    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])
    
    train_set = hf.create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    
    train_set, valid_set = hf.split_into_two_sets(train_set, first_set_fraction=1-0.2)
    
    iterator = hf.CropsFromTrialsIterator(batch_size=60, input_time_length=1000, n_preds_per_input=4)
    
    return iterator, train_set, valid_set

def shallow_brain_model(features, mode):
    print features
    
    conv1 = tf.layers.conv2d(inputs=features, filters = 40, kernel_size=(1, 25), strides=1, activation=None)
    print conv1
    
    conv2 = tf.layers.conv2d(inputs=conv1, filters = 40, kernel_size=(22, 1), strides=1, activation=None)
    batch2 = tf.layers.batch_normalization(conv2, momentum=0.1)
    activation2 = tf.nn.elu(batch2)
    pool2 = tf.layers.max_pooling2d(inputs=activation2, pool_size=(1, 80), strides=15)
    dropout2 = tf.layers.dropout(pool2, 0.5, training=mode)
    print dropout2
    
    conv3 = tf.layers.conv2d(inputs=dropout2, filters = 4, kernel_size=(1, 60), strides=1, activation=None)
    print conv3
    
    logits = tf.reshape(conv3, [-1, 4])
    print logits
    
    return logits

def train_loop(filename, labels_filename):
    epoch_number = 20

    #batch_size = 60
    #data, labels = load_data(filename)
    #data *= 10**6
    #sc = StandardScaler()
    #data = sc.fit_transform(data.reshape((288, -1)))
    #data = data.reshape((288, 22, 1126, 1))
    #test_data = data[:batch_size, :, :, :]
    #test_labels = labels[:batch_size]
    #data = data[2*batch_size:, :, :, :]
    #labels = labels[2*batch_size:]
    #num_batches = data.shape[0] / batch_size
    
    iterator, train_set, valid_set = BCI_load_data(filename, labels_filename)
    
    #d = tf.placeholder(tf.float32, [None, 22, 1126, 1])
    d = tf.placeholder(tf.float32, shape=[None, None, None, 1])
    #l = tf.placeholder(tf.int32, [None])
    l = tf.placeholder(tf.int32, shape=[None])
    m = tf.placeholder(tf.bool)
    
    model = shallow_brain_model(d, m)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=l, logits=model)
    
    lr = tf.placeholder(tf.float32)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
         
        saver = tf.train.Saver()
        
# =============================================================================
#         try:
#             saver.restore(sess, "../checkpoints/model_3_epoch_99.ckpt")
#         except Exception as e:
#             print (e)
# =============================================================================
        
        print("{} Start training...".format(datetime.now()))
        
        for epoch in range(0, epoch_number):
            all_loss = []
            all_targets = []
            all_pred_labels = []
            test_pred = []
            all_test_labels = []
                
            
            batch_generator = iterator.get_batches(train_set, shuffle=True)
            batch_generator_2 = iterator.get_batches(valid_set, shuffle=True)

            #for step in range(num_batches):
            for inputs, targets in batch_generator:
                #train_data = data[step*batch_size: (step+1)*batch_size]
                #train_label = labels[step*batch_size: (step+1)*batch_size]
                train_data = inputs
                train_label = targets

                _, ret_loss, logits = sess.run([train_op, loss, model], feed_dict={d: train_data, m:True, lr:0.005, l:train_label})

                prediction = tf.argmax(logits, 1)
                all_pred_labels.extend(prediction.eval())
            
                all_loss.append(ret_loss)
                all_targets.extend(train_label)
                
            #accuracy = np.sum(np.equal(all_pred_labels, labels)) / float(len(all_pred_labels))
            accuracy = np.sum(np.equal(all_pred_labels, all_targets)) / float(len(all_pred_labels))
            
            print("Epoch {}, Loss {:.6f}, accuracy {}, Kappa {}".format(epoch, np.mean(all_loss), accuracy, cohen_kappa_score(all_pred_labels, all_targets)))
          
            #for step in range(1):
            for inputs, targets in batch_generator_2:
                #t_data = test_data[step*batch_size: (step+1)*batch_size]
                #test_label = labels[step*batch_size: (step+1)*batch_size]
                test_data = inputs
                test_labels = targets

                ret_loss, logits = sess.run([loss, model], feed_dict={d: test_data, m: False, l:test_labels})

                prediction = tf.argmax(logits, 1)
                test_pred.extend(prediction.eval())
                all_test_labels.extend(test_labels)
                
            #accuracy = np.sum(np.equal(test_pred, test_labels)) / float(len(test_pred))
            accuracy = np.sum(np.equal(test_pred, all_test_labels)) / float(len(test_pred))
            
            print("Loss {}, Accuracy: {}, Kappa: {}".format(ret_loss, accuracy, cohen_kappa_score(test_pred, all_test_labels)))

        print("{} Done training...".format(datetime.now()))
        
        saver.save(sess, "../checkpoints/model_epoch_" + str(epoch) + ".ckpt" )

def main(argv):
    file_path = "../data/BCICIV_2a_gdf/A03T.gdf"
    labesl_file_path = "../data/true_labels/A03T.mat"
    
    train_loop(file_path, labesl_file_path)
    
if __name__ == "__main__":
    tf.app.run()