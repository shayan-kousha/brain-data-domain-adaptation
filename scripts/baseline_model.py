#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 17:01:57 2018

@author: shayan
from numpy.random import RandomState
"""
import pandas as pd
from collections import OrderedDict, Counter
import scipy
import scipy.signal
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import cohen_kappa_score
from mne import Epochs, pick_types, find_events
from mne.io import concatenate_raws, read_raw_edf, Raw, find_edf_events
from scipy.io import matlab
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_data(file_path, test_file_path):
    event_id = dict(left_hand=769, right_hand=770, both_feet=771, tongue=772)
    event_id_2 = dict(cue_unknown = 783)
        
    #tmin, tmax = -1., 4.
    tmin, tmax = -0.5, 4
    
    
    raw = read_raw_edf(file_path, eog= [22, 23, 24], stim_channel="auto", preload=True)
    raw.filter(0.5, 38.)
    events = find_edf_events(raw)
    events = np.stack((events[1], events[3], events[2]), 1).astype(np.int64)
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
    epochs_train = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    #epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    #labels = epochs_train.events[:, -1] - 769
    labels = epochs_train.events[:, -1] - 769
    
# =============================================================================
#     
#     raw = read_raw_edf(test_file_path[0], eog= [22, 23, 24], stim_channel="auto", preload=True)
#     raw.filter(0.5, 100.)
#     events = find_edf_events(raw)
#     events = np.stack((events[1], events[3], events[2]), 1).astype(np.int64)
#     picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
#                        exclude='bads')
#     epochs_train_2 = Epochs(raw, events, event_id_2, tmin, tmax, proj=True, picks=picks,
#                     baseline=None, preload=True)
#     #epochs_train_2 = epochs.copy().crop(tmin=1., tmax=2.)
#     
#     labels_2 = matlab.loadmat(test_file_path[1])['classlabel']
#     labels_2 = labels_2.squeeze()
#     labels_2 -= 1
#         
#     data_train = np.concatenate((epochs_train.get_data()[:, :, :, np.newaxis], epochs_train_2.get_data()[:, :, :, np.newaxis]))
#     labels_train = np.concatenate((labels, labels_2))
#     
#     indices = np.arange(data_train.shape[0])   
#     np.random.shuffle(indices)
#     
#     return data_train[indices], labels_train[indices]
# =============================================================================
    
    indices = np.arange(labels.shape[0])
    #np.random.shuffle(indices)
    
    return epochs_train.get_data()[:, :, :, np.newaxis][indices], labels[indices]
    
    
    
    
    
# =============================================================================
# =============================================================================
# =============================================================================
# # # def load_data(file_path, mode):
# # #     if mode == 'test':
# # #         event_id = dict(cue_unknown = 783)
# # #         
# # #         #event_id = dict(left_hand=769, right_hand=770, both_feet=771, tongue=772)
# # #         gdf_files = [file_path[0]]
# # #     else:
# # #         event_id = dict(left_hand=769, right_hand=770, both_feet=771, tongue=772)
# # #         gdf_files = file_path
# # #         
# # #     tmin, tmax = -1., 4.
# # # 
# # #     all_data = []
# # #     all_labels = []
# # #     
# # #     for i in range(len(gdf_files) - 1):
# # #     
# # #         raw = read_raw_edf(gdf_files[i], eog= [22, 23, 24], stim_channel="auto")
# # # 
# # #         events = find_edf_events(raw)
# # #         events = np.stack((events[1], events[3], events[2]), 1).astype(np.int64)
# # #         
# # #         picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
# # #                            exclude='bads')
# # #         epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
# # #                         baseline=None, preload=True)
# # #         epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
# # #     
# # #         if mode == 'test':
# # #             labels = matlab.loadmat(file_path[1])['classlabel']
# # #             labels = labels.squeeze()
# # #             labels -= 1
# # #         else:
# # #             labels = epochs_train.events[:, -1] - 769
# # #     
# # #         all_data.extend(epochs_train.get_data()[:, :, :, np.newaxis])
# # #         all_labels.extend(labels)
# # #         
# # #     all_data = np.array(all_data)
# # #     all_labels = np.array(all_labels)
# # #     
# # #     indices = np.arange(all_data.shape[0])   
# # #     np.random.shuffle(indices)
# # #     
# # #     ####################################################
# # #     
# # #     raw = read_raw_edf(gdf_files[-1], eog= [22, 23, 24], stim_channel="auto")
# # # 
# # #     events = find_edf_events(raw)
# # #     events = np.stack((events[1], events[3], events[2]), 1).astype(np.int64)
# # #     
# # #     picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
# # #                        exclude='bads')
# # #     epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
# # #                     baseline=None, preload=True)
# # #     epochs_test = epochs.copy().crop(tmin=1., tmax=2.)
# # # 
# # #     if mode == 'test':
# # #         labels = matlab.loadmat(file_path[1])['classlabel']
# # #         labels = labels.squeeze()
# # #         labels -= 1
# # #     else:
# # #         labels = epochs_test.events[:, -1] - 769
# # #         
# # #     indices_test = np.arange(epochs_test.get_data().shape[0])   
# # #     np.random.shuffle(indices_test)
# # #     
# # #     return all_data[indices], all_labels[indices], epochs_test.get_data()[:, :, :, np.newaxis][indices_test], labels[indices_test]
# # #     ####################################################
# # #     
# # #     #return all_data[indices], all_labels[indices]
# =============================================================================
# =============================================================================
# =============================================================================

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

# =============================================================================
# =============================================================================
# # def brain_model(features, num_classes=4):
# #     batch0 = tf.layers.batch_normalization(features)
# #     
# #     conv1 = tf.layers.conv2d(inputs=batch0, filters = 20, kernel_size=(1, 5), strides=1, activation=tf.nn.elu)
# #     batch1 = tf.layers.batch_normalization(conv1)
# #     print conv1
# #     conv2 = tf.layers.conv2d(inputs=batch1, filters = 20, kernel_size=(22, 1), strides=1, activation=tf.nn.elu)
# #     batch2 = tf.layers.batch_normalization(conv2)
# #     pool2 = tf.layers.max_pooling2d(inputs=batch2, pool_size=(1, 3), strides=2)
# #     
# #     conv3 = tf.layers.conv2d(inputs=pool2, filters = 50, kernel_size=(1, 20), strides=1, activation=tf.nn.elu)
# #     batch3 = tf.layers.batch_normalization(conv3)
# #     pool3 = tf.layers.max_pooling2d(inputs=batch3, pool_size=(1, 3), strides=2)
# # 
# #     conv4 = tf.layers.conv2d(inputs=pool3, filters = 100, kernel_size=(1, 30), strides=1, activation=tf.nn.elu)
# #     pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=(1, 3), strides=2)
# # 
# #     conv5 = tf.layers.conv2d(inputs=pool4, filters = 200, kernel_size=(1, 20), strides=1, activation=tf.nn.elu)
# #     pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=(1, 3), strides=2)
# # 
# #     conv6 = tf.layers.conv2d(inputs=pool5, filters = 4, kernel_size=(1, 34), strides=1, activation=None)
# # 
# #     logits = tf.reshape(conv6, [-1, 4])
# #     
# #     return logits
# =============================================================================
# =============================================================================

def train_loop(file_path, test_file_path, mode):
    data, labels = load_data(file_path, test_file_path)
    data *= 10**6
    
    epoch_number = 200
    batch_size = 60
    
    
    #sc = StandardScaler()
    #data = sc.fit_transform(data.reshape((288, -1)))
    #data = data.reshape((288, 22, 1126, 1))
    
    #test_data = data[:batch_size, :, :, :]
    #test_labels = labels[:batch_size]
    
    #data = data[2*batch_size:, :, :, :]
    #labels = labels[2*batch_size:]
    
    
    #num_batches = data.shape[0] / batch_size
    
    
    
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
            
        if mode == 'train':
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
            
            saver.save(sess, "../checkpoints/model_3_epoch_" + str(epoch) + ".ckpt" )
        else:
            print("{} Start testing...".format(datetime.now()))
            
            all_pred_labels = []
            
            for step in range(num_batches):
                test_data = data[step*batch_size: (step+1)*batch_size]
                # test_label = labels[step*batch_size: (step+1)*batch_size]
            
                logits = sess.run(model, feed_dict={d: test_data})
                
                prediction = tf.argmax(logits, 1)
                all_pred_labels.extend(prediction.eval())
                
            accuracy = np.sum(np.equal(all_pred_labels, labels)) / float(len(all_pred_labels))
            
            print("Accuracy: {}, Kappa: {}".format(accuracy, cohen_kappa_score(all_pred_labels, labels)))

def main(argv):
    test_file_path = ["../data/BCICIV_2a_gdf/A03E.gdf", "../data/true_labels/A03E.mat"]
    file_path = "../data/BCICIV_2a_gdf/A03T.gdf"
    mode = 'train'
    train_loop(file_path, test_file_path, mode)
    
if __name__ == "__main__":
    tf.app.run()
    
    
    
    
########################
train_loader = BCICompetition4Set2A("../data/BCICIV_2a_gdf/A03T.gdf", labels_filename="../data/true_labels/A03T.mat")
    
train_cnt = train_loader.load()

 train_cnt = train_cnt.drop_channels(['STI 014', 'EOG-left',
                                     'EOG-central', 'EOG-right'])
assert len(train_cnt.ch_names) == 22
# lets convert to millvolt for numerical stability of next operations
train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
train_cnt = mne_apply(
    lambda a: bandpass_cnt(a, 0, 38.0, train_cnt.info['sfreq'],
                           filt_order=3,
                           axis=1), train_cnt)
train_cnt = mne_apply(
    lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                              init_block_size=1000,
                                              eps=1e-4).T,
    train_cnt)

marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                          ('Foot', [3]), ('Tongue', [4])])
ival = [-500, 4000]
train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)

   train_set, valid_set = split_into_two_sets(
    train_set, first_set_fraction=1-0.2)

iterator = CropsFromTrialsIterator(batch_size=60,
                               input_time_length=1000,
                               n_preds_per_input=4)

batch_generator = iterator.get_batches(train_set, shuffle=True)
batch_generator_2 = iterator.get_batches(valid_set, shuffle=True)

log = logging.getLogger(__name__)