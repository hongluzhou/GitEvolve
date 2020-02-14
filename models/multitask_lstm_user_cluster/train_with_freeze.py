#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:38:29 2019

@author: zhouhonglu
"""
from models.multitask_lstm_user_cluster.utils import categorical_focal_loss
from models.multitask_lstm_user_cluster.utils import set_logger
from models.multitask_lstm_user_cluster.utils import print_and_log
from models.multitask_lstm_user_cluster.utils import time_delay_normalization
from models.multitask_lstm_user_cluster.utils import load_jsongz_2_dataframe
from models.multitask_lstm_user_cluster.utils import (
        insert_no_event_for_a_chain_new)
from models.multitask_lstm_user_cluster.utils import get_time_delay
from models.multitask_lstm_user_cluster.utils import utc_timestamp
from models.multitask_lstm_user_cluster.utils import weighted_et_bce
from models.multitask_lstm_user_cluster.utils import weighted_uc_bce

import os
import pdb
import time
from datetime import datetime as dt
import json
import pandas as pd
import numpy as np
import pickle

import keras
from keras.optimizers import Adam
from keras import metrics
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout
# from keras.layers.normalization import BatchNormalization
from keras.losses import logcosh, binary_crossentropy
from keras import losses
from keras.utils import multi_gpu_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
# from keras.callbacks import EarlyStopping
# from keras.callbacks import TensorBoard
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

pd.set_option('display.max_columns', None)


def train(config, process_create_dataset=True):
    if process_create_dataset:
        trainset_valiset_creation(config)

    train_models(config)
    return


def train_models(config):
    # obtain x_dim
    x_dim = config['x_dim']

    if not os.path.exists(os.path.join(
            config['exp_save_dir'], "models", "vis")):
        os.makedirs(os.path.join(
                config['exp_save_dir'], "models", 'vis'))

    logger = set_logger(os.path.join(config['exp_save_dir'], "models",
                                     'train_model_architecture_' +
                                     dt.now().strftime("%Y-%m-%dT%H-%M-%SZ") +
                                     '.log'))
    print_and_log(logger, "x_dim: {}".format(x_dim))

    # LTSM model architecture
    input_sequences = Input(shape=(config['window_size'], x_dim),
                            name='input_sequences')
    lstm_1 = LSTM(250, return_sequences=True, name="lstm_1")(
                    input_sequences)
    lstm_2 = LSTM(150, name="lstm_2")(lstm_1)

    # branching event type
    et_1 = Dense(128, activation='relu', name="et_1")(lstm_2)
    et_2 = Dropout(0.5, name="et_2")(et_1)
    et_3 = Dense(64, activation='relu', name="et_3")(et_2)
    et_4 = Dropout(0.5, name="et_4")(et_3)
    event_type_output = Dense(len(config['eventtype_2_id']),
                              activation='sigmoid',
                              name="event_type_output")(et_4)

    # branching time delay
    td_1 = Dense(128, activation='relu', name="td_1")(lstm_2)
    td_2 = Dropout(0.5, name="td_2")(td_1)
    td_3 = Dense(64, activation='relu', name="td_3")(td_2)
    td_4 = Dropout(0.5, name="td_4")(td_3)
    time_delay_output = Dense(1, activation='linear',
                              name="time_delay_output")(td_4)

    # branching user cluster
    uc_1 = Dense(128, activation='relu', name="uc_1")(lstm_2)
    uc_2 = Dropout(0.5, name="uc_2")(uc_1)
    uc_3 = Dense(64, activation='relu', name="uc_3")(uc_2)
    uc_4 = Dropout(0.5, name="uc_4")(uc_3)
    user_cluster_output = Dense(config['dim_user_cluster_one_hot'],
                                activation='sigmoid',
                                name="user_cluster_output")(uc_4)
    # model
    model = Model(inputs=input_sequences,
                  outputs=[event_type_output,
                           time_delay_output,
                           user_cluster_output])

    model.summary(print_fn=logger.info)

    print(model.summary())

    model = multi_gpu_model(model, gpus=config['num_gpu'])

    # get partition and labels
    with open(os.path.join(config['exp_save_dir'], "dataset",
                           'partition.json'), 'r') as f:
        partition = json.load(f)
    with open(os.path.join(config['exp_save_dir'], "dataset",
                           'labels.json'), 'r') as f:
        labels = json.load(f)

    # train val generator
    training_generator = DataGenerator_ET_TD_UC_one_hot(
            os.path.join(config['exp_save_dir'], "dataset"),
            partition['train'],
            labels, batch_size=config['batch_size'],
            dim=x_dim,
            window_size=config['window_size'],
            et_classes=len(config['eventtype_2_id']),
            uc_classes=config['dim_user_cluster_one_hot'],
            shuffle=config['generator_shuffle'])
    validation_generator = DataGenerator_ET_TD_UC_one_hot(
            os.path.join(config['exp_save_dir'], "dataset"),
            partition['validation'],
            labels, batch_size=config['batch_size'],
            dim=x_dim,
            window_size=config['window_size'],
            et_classes=len(config['eventtype_2_id']),
            uc_classes=config['dim_user_cluster_one_hot'],
            shuffle=config['generator_shuffle'])

    # callback
    # if not os.path.exists(os.path.join(
    #         config['exp_save_dir'], "models", "vis")):
    #     os.makedirs(os.path.join(
    #             config['exp_save_dir'], "models", 'vis'))

    # TensorBoard_callback = TensorBoard(
    #         log_dir=os.path.join(config['exp_save_dir'], "models", 'vis'),
    #         histogram_freq=0,
    #         write_graph=True,
    #         write_images=True,
    #         write_grads=False)
    """
    Keras TensorBoard Reference:
        https://keras.io/callbacks/#tensorboard

    launch: tensorboard --logdir=/full_path_to_your_logs
    """

    callbacks = [  # EarlyStopping(monitor='val_loss', patience=50),
                 ModelCheckpoint(
                         filepath=os.path.join(config['exp_save_dir'],
                                               "models",
                                               'model.hdf5'),
                         monitor='val_loss',
                         verbose=2)]

    # save train confg in case testing need it
    with open(os.path.join(config['exp_save_dir'], "models",
                           'train_config.pickle'), 'wb') as f:
        pickle.dump(config, f)
    print("{} saved!".format('train_config.pickle'))

    # model_history
    model_history = dict()

    # start training
    for epoch in range(config['num_epochs']):
        print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!  "
              "Epoch {}/{}    {}".format(
                      epoch+1, config['num_epochs'],
                      "---> train all branches..."))

        trainable_count = int(
            np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
        non_trainable_count = int(
            np.sum([K.count_params(p) for p in set(
                    model.non_trainable_weights)]))

        print('Total params: {:,}'.format(
                trainable_count + non_trainable_count))
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))

        if (config['use_user_cluster_one_hot']) and (
                not config['use_user_cluster_minmax']):
            if config['et_loss'] == 'focalloss':
                model.compile(optimizer=Adam(lr=config['lr'],
                                             decay=config['decay'],
                                             amsgrad=config['amsgrad']),
                              loss=[categorical_focal_loss(
                                      gamma=config['focalloss_gamma'],
                                      alpha=config['focalloss_alpha']),
                                    logcosh,
                                    binary_crossentropy],
                              loss_weights=config['loss_weights'],
                              metrics={'event_type_output': (
                                              metrics.categorical_accuracy),
                                       'time_delay_output': (
                                               losses.mean_squared_error),
                                       'user_cluster_output': (
                                                metrics.categorical_accuracy)
                                       })
            elif config['et_loss'] == 'bce':
                model.compile(optimizer='adam',
                              loss=[binary_crossentropy,
                                    logcosh,
                                    binary_crossentropy],
                              loss_weights=config['loss_weights'],
                              metrics={'event_type_output': (
                                               metrics.categorical_accuracy),
                                       'time_delay_output': (
                                               losses.mean_squared_error),
                                       'user_cluster_output': (
                                               metrics.categorical_accuracy)
                                       })
            else:
                print("event type loss undefined!")
                pdb.set_trace()
        else:
            model.compile(optimizer='adam',
                          loss=[categorical_focal_loss(
                                  gamma=config['focalloss_gamma'],
                                  alpha=config['focalloss_alpha']),
                                logcosh,
                                logcosh],
                          loss_weights=config['loss_weights'],
                          metrics={'event_type_output': (
                                           metrics.categorical_accuracy),
                                   'time_delay_output': (
                                           losses.mean_squared_error),
                                   'user_cluster_output': (
                                           losses.mean_squared_error)})

        history = model.fit_generator(
                generator=training_generator,
                epochs=1,
                callbacks=callbacks,
                validation_data=validation_generator,
                use_multiprocessing=True,
                workers=config['multiprocessing_cpu'],
                shuffle=True)
        """ Whether to shuffle the order of the batches
        at the beginning of each epoch.
        Only used with instances of Sequence (keras.utils.Sequence).
        Has no effect when steps_per_epoch is not None.

        Basically, no effect here.

        https://stackoverflow.com/questions/49027174/
            what-does-shuffle-do-in-fit-generator-in-keras
        """

        model.save(os.path.join(config['exp_save_dir'], "models",
                                'model-{}.hdf5'.format(epoch+1)))
        print("model-{}.hdf5 saved!".format(epoch+1))

        if len(model_history) == 0:
            model_history = history.history.copy()
        else:
            for key in history.history:
                model_history[key] += history.history[key]

        with open(os.path.join(config['exp_save_dir'], "models",
                               'history-{}.pickle'.format(epoch+1)),
                  'wb') as f:
            pickle.dump(model_history, f)
        print("history-{}.pickle saved!".format(epoch+1))

    with open(os.path.join(config['exp_save_dir'], "models",
                           'history.pickle'),
              'wb') as f:
        pickle.dump(model_history, f)
    print("history.pickle saved!")

    return


def trainset_valiset_creation(config):
    # preparation
    create_dataset_start = time.time()
    if not os.path.exists(os.path.join(config['exp_save_dir'], "dataset")):
        os.makedirs(os.path.join(config['exp_save_dir'], "dataset"))

    logger = set_logger(os.path.join(config['exp_save_dir'],
                                     'trainset_valiset_creation_' +
                                     dt.now().strftime("%Y-%m-%dT%H-%M-%SZ") +
                                     '.log'))

    print_and_log(logger, '{}'.format(
                'trainset and valiset creation...'))

    eventtype_2_id = config['eventtype_2_id']
    # id_2_eventtype = dict(zip(eventtype_2_id.values(),
    #                           eventtype_2_id.keys()))

    with open(config['github_event_repos_path'], 'rb') as f:
        github_event_repos = pickle.load(f)
    github_event_repos_set = set(
            [repo[:22]+'-'+repo[23:] for repo in github_event_repos])

    train_start = utc_timestamp(config['train_period']['start'])
    train_end = utc_timestamp(config['train_period']['end'])
    vali_start = utc_timestamp(config['vali_period']['start'])
    vali_end = utc_timestamp(config['vali_period']['end'])

    # read cascade file to get whole dataset
    print_and_log(logger, "read cascade files...")

    x_dim = config['x_dim']
    if config['use_repo_embedding']:
        with open(config['embed_vec_path'], 'rb') as f:
            embed_vec_all = pickle.load(f)
    if config['use_repo_profile_features']:
        with open(config['repo_profile_feat_path'], 'r') as f:
            repo_profile_raw = json.load(f)
        repo_profile_feat = dict()
        for repo in repo_profile_raw:
            this_repo_profile_feat = []
            this_repo_profile_feat.append(
                    time_delay_normalization(
                            repo_profile_raw[repo]['language'][0],
                            config['time_delay_normalization_func']))
            this_repo_profile_feat += repo_profile_raw[repo]['user_type']
            repo_profile_feat[repo] = this_repo_profile_feat
    if config['use_repo_idx_features']:
        if config['load_repo_path'] == 'null':
            with open(os.path.join(config['root_dir'], 'data',
                                   'unique_repo_train_vali.json'), 'r') as f:
                unique_repo_train_vali = json.load(f)

            unique_repo_list = list(unique_repo_train_vali.keys())

            load_repo_train_vali = dict()
            for i in range(len(unique_repo_list)):
                repo = unique_repo_list[i]
                load_repo_train_vali[repo] = [
                        time_delay_normalization(
                                i, config['time_delay_normalization_func'])]

            with open(os.path.join(config['root_dir'], 'data',
                                   'load_repo_train_vali.pickle'), 'wb') as f:
                pickle.dump(load_repo_train_vali, f)
        else:
            with open(config['load_repo_path'], 'rb') as f:
                load_repo_train_vali = pickle.load(f)
    if config['use_repo_activity_features']:
        max_sum_act_feat = 0
        max_sum_act_feat_vec = None
        with open(config['repo_activity_feat_path'], 'rb') as f:
            repo_act_feat = pickle.load(f)
        for repo in repo_act_feat:
            for cluster in repo_act_feat[repo]:
                this_repo_cluster_feat = []
                for feat_name in repo_act_feat[repo][cluster]:
                    this_repo_cluster_feat += repo_act_feat[
                            repo][cluster][feat_name]
                repo_act_feat[repo][cluster] = this_repo_cluster_feat
                if sum(this_repo_cluster_feat) > max_sum_act_feat:
                    max_sum_act_feat = sum(this_repo_cluster_feat)
                    max_sum_act_feat_vec = this_repo_cluster_feat
        print_and_log(logger, "max_sum_act_feat: {}".format(
                max_sum_act_feat))
        print_and_log(logger, "max_sum_act_feat_vec: {}".format(
                max_sum_act_feat_vec))
    if config['use_user_profile_features']:
        with open(config['user_profile_feat_path'], 'r') as f:
            user_profile_feat = json.load(f)
    if config['use_user_activity_features']:
        with open(config['user_activity_feat_path'], 'r') as f:
            user_act_feat = json.load(f)
    if config['use_user_cluster_one_hot']:
        with open(config['user_cluster_path'], 'r') as f:
            user_clusters = json.load(f)
        empty_no_event_user_cluster = max(user_clusters.values()) + 1
        user_clusters[config['empty_user']] = empty_no_event_user_cluster
        user_clusters['no_event_user'] = empty_no_event_user_cluster
    if config['use_user_cluster_minmax']:
        with open(config['user_cluster_path'], 'r') as f:
            user_clusters = json.load(f)
        max_cluster_id = max(user_clusters.values())
        min_cluster_id = min(user_clusters.values())
        empty_no_event_user_cluster = max_cluster_id + 1
        max_cluster_id = empty_no_event_user_cluster
        max_minus_min = max_cluster_id - min_cluster_id
        if min_cluster_id != 0:
            print("min cluster id is not 0! Need to examine code!")
            pdb.set_trace()
        user_clusters[config['empty_user']] = empty_no_event_user_cluster
        user_clusters['no_event_user'] = empty_no_event_user_cluster
    if config['use_cluster_profile_features']:
        max_sum_profile_feat = 0
        max_sum_profile_feat_vec = None
        with open(config['cluster_profile_feat_path'], 'rb') as f:
            cluster_profile_feat = pickle.load(f)
        for cluster in cluster_profile_feat:
            this_cluster_feat = []
            for feat_name in cluster_profile_feat[cluster]:
                # if feat_name == "geolocation" or feat_name == "user_type":
                #     continue
                this_cluster_feat += cluster_profile_feat[cluster][feat_name]
            cluster_profile_feat[cluster] = this_cluster_feat
            if sum(this_cluster_feat) > max_sum_profile_feat:
                max_sum_profile_feat = sum(this_cluster_feat)
                max_sum_profile_feat_vec = this_cluster_feat
        cluster_profile_feat[empty_no_event_user_cluster] = [0] * config[
                'dim_cluster_profile_features']
        print_and_log(logger, "max_sum_profile_feat: {}".format(
                max_sum_profile_feat))
        print_and_log(logger, "max_sum_profile_feat_vec: {}".format(
                max_sum_profile_feat_vec))
    if config['use_cluster_activity_features']:
        max_sum_act_feat = 0
        max_sum_act_feat_vec = None
        with open(config['cluster_activity_feat_path'], 'rb') as f:
            cluster_act_feat = pickle.load(f)
        for cluster in cluster_act_feat:
            for repo in cluster_act_feat[cluster]:
                this_cluster_repo_feat = []
                for feat_name in cluster_act_feat[cluster][repo]:
                    this_cluster_repo_feat += cluster_act_feat[
                            cluster][repo][feat_name]
                cluster_act_feat[cluster][repo] = this_cluster_repo_feat
                if sum(this_cluster_repo_feat) > max_sum_act_feat:
                    max_sum_act_feat = sum(this_cluster_repo_feat)
                    max_sum_act_feat_vec = this_cluster_repo_feat
        print_and_log(logger, "max_sum_act_feat: {}".format(
                max_sum_act_feat))
        print_and_log(logger, "max_sum_act_feat_vec: {}".format(
                max_sum_act_feat_vec))

    print_and_log(logger, "x_dim: {}".format(x_dim))
    # pdb.set_trace()

    partition = dict()
    partition['train'] = []
    partition['validation'] = []
    labels = dict()
    labels['event_type'] = dict()
    labels['time_delay'] = dict()
    labels['user_cluster'] = dict()

    repo2sample = dict()

    repo_list = [f[:-8] for f in os.listdir(
                config['cascade_dir']) if '.json.gz' in f]

    sample_id = 0

    user_has_no_cluster = set()
    unique_repo_train_vali = {}
    unique_user_train_vali = {}

    # each_repo_chain_length = dict()
    # each_user_total_event_count = dict()
    # each_cluster_total_event_count = dict()
    # each_event_type_total_count = dict()

    for repo_idx in range(len(repo_list)):
        repo = repo_list[repo_idx]
        print('processing {}, {}/{}'.format(
                round(repo_idx/len(repo_list), 2),
                repo_idx,
                len(repo_list)),
              end='\r')

        if (repo in config['repos_to_ignore']) or (
                repo in github_event_repos_set):
            # it is a event repo or repo should be ignore
            continue

        repo2sample[repo] = dict()
        repo2sample[repo]['train'] = list()
        repo2sample[repo]['validation'] = list()

        one_chain_pd = load_jsongz_2_dataframe(
                os.path.join(config['cascade_dir'], repo + '.json.gz'))
        if len(one_chain_pd.loc[
                (one_chain_pd['nodeTime'] >= train_start) &
                (one_chain_pd['nodeTime'] <= train_end)]) == 0:
            # this repo has no events in training
            continue

#        if repo == 'Pg6sypDT02F199RR24XAGw-FNnptRvRMDUIs9HwCqbS6A':
#            sim_start = utc_timestamp(config['sim_period']['start'])
#            sim_end = utc_timestamp(config['sim_period']['end'])
#
#            tem_a = one_chain_pd.loc[
#                    (one_chain_pd['nodeTime'] >= train_start) &
#                    (one_chain_pd['nodeTime'] <= vali_end)]
#            tem_a_time = []
#            tem_a_time += [time for time in tem_a['nodeTime']]
#
#            tem_b = one_chain_pd.loc[
#                    (one_chain_pd['nodeTime'] >= sim_start) &
#                    (one_chain_pd['nodeTime'] <= sim_end)]
#            tem_b_time = []
#            tem_b_time += [time for time in tem_b['nodeTime']]
#            pdb.set_trace()

        #############################

        # one_chain_event = []
        # one_chain_user = []
        # event sequence
        # one_chain_event += [event for event in one_chain_pd['actionType']]
        # one_chain_user += [user for user in one_chain_pd['nodeUserID']]

        # each_repo_chain_length[repo] = len(one_chain_event)

        # for user in one_chain_user:
        #     try:
        #         each_user_total_event_count[user] += 1
        #     except KeyError:
        #         each_user_total_event_count[user] = 1

        # for event in one_chain_event:
        #     try:
        #         each_event_type_total_count[event] += 1
        #     except KeyError:
        #         each_event_type_total_count[event] = 1

        # continue

        #############################

        one_chain_pd = one_chain_pd.loc[
                    (one_chain_pd['nodeTime'] >= train_start) &
                    (one_chain_pd['nodeTime'] <= vali_end)]

        one_chain_pd = one_chain_pd.sort_values(by=['nodeTime'])

        one_chain_event = []
        one_chain_time = []
        one_chain_user = []
        # padding event
        for i in range(config['window_size']):
            one_chain_event.append(config['empty_event_type'])
            one_chain_time.append(config['empty_time_delay'])
            one_chain_user.append(config['empty_user'])
        # <soc>
        one_chain_event.append(eventtype_2_id['<soc>'])
        one_chain_time.append(config['empty_time_delay'])
        one_chain_user.append(config['empty_user'])

        # event sequence
        one_chain_event += [eventtype_2_id[
                event] for event in one_chain_pd['actionType']]
        one_chain_time += [time for time in one_chain_pd['nodeTime']]
        one_chain_user += [user for user in one_chain_pd['nodeUserID']]

        (one_chain_event_new, one_chain_time_new, one_chain_user_new) = \
            insert_no_event_for_a_chain_new(config,
                                            one_chain_event, one_chain_time,
                                            one_chain_user, vali_end+1)
        # if one_chain_event_new != one_chain_event:
        #     pdb.set_trace()

        one_chain_event = one_chain_event_new
        one_chain_time = one_chain_time_new
        one_chain_user = one_chain_user_new

        """
        one_chain_event = one_chain_event_new[21:]
        one_chain_time = one_chain_time_new[21:]
        one_chain_user = one_chain_user_new[21:]
        one_chain_cluster = [user_clusters[user] for user in one_chain_user]

        for cluster in one_chain_cluster:
            try:
                each_cluster_total_event_count[cluster] += 1
            except KeyError:
                each_cluster_total_event_count[cluster] = 1

        for event in one_chain_event:
            try:
                each_event_type_total_count[event] += 1
            except KeyError:
                each_event_type_total_count[event] = 1

        continue
        """

        # calculate time delay sequence
        one_chain_time_delay = []
        for i in range(len(one_chain_time)):
            if (one_chain_event[i] == config['empty_event_type'] or
                    one_chain_event[i] == eventtype_2_id['<soc>']):
                one_chain_time_delay.append(config['empty_time_delay'])
            elif one_chain_event[i-1] == eventtype_2_id['<soc>']:
                one_chain_time_delay.append(config['empty_time_delay'])
            else:
                time_delay = get_time_delay(one_chain_time[i-1],
                                            one_chain_time[i],
                                            'float')[1]
                if config['time_delay_normalization_func'] is not None:
                    time_delay = time_delay_normalization(
                            time_delay,
                            config['time_delay_normalization_func'])
                one_chain_time_delay.append(time_delay)

        # get the unique repos users in the training cascades
        unique_repo_train_vali[repo] = []
        for user in one_chain_pd['nodeUserID']:
            unique_user_train_vali[user] = []

        # for each sample
        for i in range(config['window_size'], len(one_chain_event)):
            sample_id += 1
            ID = 'id-' + str(sample_id)
            # print(ID)
            # pdb.set_trace()

            time_sample_outputevent = one_chain_time[i]
            event_sample_outputevent = one_chain_event[i]

            # if time_sample_outputevent in training period:
            # add this sample to trainset
            if event_sample_outputevent == config['empty_event_type'] or (
                    event_sample_outputevent == eventtype_2_id['<soc>']):
                continue
            if one_chain_event[i-1] == eventtype_2_id['<soc>']:
                continue
            if not ((time_sample_outputevent >= train_start) and (
                    time_sample_outputevent <= (vali_end))):
                print("should not happen")
                pdb.set_trace()

            input_event_type = \
                one_chain_event[i-config['window_size']:i]
            input_time = one_chain_time[i-config['window_size']:i]
            input_time_delay = \
                one_chain_time_delay[i-config['window_size']:i]
            input_user = one_chain_user[i-config['window_size']:i]
            input_cluster = []
            for user in input_user:
                try:
                    input_cluster.append(user_clusters[user])
                except KeyError:
                    user_has_no_cluster.add(user)
                    input_cluster.append(user_clusters['no_event_user'])

            output_event_type = \
                one_chain_event[i]
            output_time_delay = \
                one_chain_time_delay[i]
            output_user = one_chain_user[i]
            try:
                output_cluster = user_clusters[output_user]
            except KeyError:
                user_has_no_cluster.add(output_user)
                output_cluster = user_clusters['no_event_user']

            """
            if (config['use_user_cluster_one_hot']) and (
                    not config['use_user_cluster_minmax']):
                try:
                    output_cluster = user_clusters[output_user]
                except KeyError:
                    user_has_no_cluster.add(output_user)
                    output_cluster = user_clusters['no_event_user']
            else:
                try:
                    output_cluster = (
                            user_clusters[output_user] / max_minus_min)
                except KeyError:
                    user_has_no_cluster.add(output_user)
                    output_cluster = (
                            user_clusters[
                                    'no_event_user'] / max_minus_min)
            """

            # initialize input vector, and output vector for this sample
            x_vec = []

            # load repo embeding vector
            if config['use_repo_embedding']:
                try:
                    embed_vec = np.array(
                            embed_vec_all[repo[:22] + '/' + repo[23:]])
                except KeyError:
                    print_and_log(logger, "Could not find "
                                  "embedding vector for {}!".format(
                                          repo[:22] + '/' + repo[23:]))
                    pdb.set_trace()

            # input feature vector
            for j in range(config['window_size']):  # for each event node
                x_j = []

                if config['use_repo_embedding']:
                    x_j += list(embed_vec)

                if config['use_repo_profile_features']:
                    try:
                        x_j += repo_profile_feat[repo]
                    except KeyError:
                        x_j += [0] * config['dim_repo_profile_features']

                if config['use_repo_idx_features']:
                    try:
                        x_j += load_repo_train_vali[repo]
                    except KeyError:
                        x_j += [0]

                if config['use_repo_activity_features']:
                    if input_cluster[j] == empty_no_event_user_cluster:
                        x_j += [0] * config[
                                'dim_repo_activity_features']
                    else:
                        try:
                            repo_thiscluster_act_feat = repo_act_feat[
                                        repo][input_cluster[j]]
                            repo_allcluster_act_feat = repo_act_feat[
                                        repo]['all_cluster']
                            x_j += repo_thiscluster_act_feat
                            x_j += repo_allcluster_act_feat
                        except KeyError:
                            x_j += [0] * config[
                                'dim_repo_activity_features']

                if config['use_user_profile_features']:
                    if input_user[j] == config['empty_user'] or (
                            input_user[j] == 'no_event_user'):
                        x_j += [0] * config['dim_user_profile_features']
                    else:
                        try:
                            x_j += user_profile_feat[input_user[j]]
                        except KeyError:
                            x_j += [0] * config[
                                    'dim_user_profile_features']

                if config['use_user_activity_features']:
                    if input_user[j] == config['empty_user'] or (
                            input_user[j] == 'no_event_user'):
                        x_j += [0] * config[
                                'dim_user_activity_features']
                    else:
                        try:
                            thisrepo_feat = \
                                user_act_feat[input_user[j]][repo]
                        except KeyError:
                            # this user-repo no event in training period
                            thisrepo_feat = \
                                [0] * int(config[
                                        'dim_user_activity_features']/2)
                        allrepo_feat = \
                            user_act_feat[input_user[j]]['all']
                        x_j += thisrepo_feat + allrepo_feat

                if config['use_event_type_one_hot']:
                    event_type_one_hot = \
                        [0] * len(config['eventtype_2_id'])
                    if input_event_type[j] != config['empty_event_type']:
                        event_type_one_hot[input_event_type[j]-1] = 1
                    x_j += event_type_one_hot

                if config['use_time_delay_features']:
                    x_j += [input_time_delay[j]]

                if config['use_user_cluster_one_hot']:
                    user_cluster_one_hot = \
                        [0] * config['dim_user_cluster_one_hot']
                    user_cluster_one_hot[input_cluster[j]] = 1

                    x_j += user_cluster_one_hot

                if config['use_user_cluster_minmax']:
                    use_user_cluster_minmax = (
                            input_cluster[j] / max_minus_min)

                    x_j += [use_user_cluster_minmax]

                if config['use_cluster_profile_features']:
                    this_cluster_profile_feat = cluster_profile_feat[
                            input_cluster[j]]
                    x_j += this_cluster_profile_feat

                if config['use_cluster_activity_features']:
                    if input_cluster[j] == empty_no_event_user_cluster:
                        x_j += [0] * config[
                                'dim_cluster_activity_features']
                    else:
                        try:
                            cluster_thisrepo_act_feat = cluster_act_feat[
                                        input_cluster[j]][repo]
                            cluster_allrepo_act_feat = cluster_act_feat[
                                        input_cluster[j]]['all_repo']
                            x_j += cluster_thisrepo_act_feat
                            x_j += cluster_allrepo_act_feat
                        except KeyError:
                            # posibility one:
                            # cluster only did first event for this repo
                            # since no time delay exist
                            # we didnot have actfeat for this clus-repo pair
                            # posibility two:
                            # cluster only did event for this repo in
                            # validation period
                            if not (input_time_delay[j] == -1 or (
                                    time_sample_outputevent > (train_end+1))):
                                print(j, input_time)
                                pdb.set_trace()
                            x_j += [0] * config[
                                'dim_cluster_activity_features']

                if len(x_j) != x_dim:
                    print("len(x_j) != x_dim")
                    pdb.set_trace()
                x_vec.append(x_j)
            if len(x_vec) != config['window_size']:
                print("len(x_vec) != config['window_size']")
                pdb.set_trace()

            if (time_sample_outputevent >= train_start) and (
                    time_sample_outputevent <= train_end):
                partition['train'].append(ID)
                labels['event_type'][ID] = output_event_type-1
                labels['time_delay'][ID] = output_time_delay
                labels['user_cluster'][ID] = output_cluster
                np.save(os.path.join(config['exp_save_dir'], "dataset",
                                     ID + '.npy'), x_vec)
                repo2sample[repo]['train'].append(ID)

            elif (time_sample_outputevent >= vali_start) and (
                    time_sample_outputevent <= vali_end):
                partition['validation'].append(ID)
                labels['event_type'][ID] = output_event_type-1
                labels['time_delay'][ID] = output_time_delay
                labels['user_cluster'][ID] = output_cluster
                np.save(os.path.join(config['exp_save_dir'], "dataset",
                                     ID + '.npy'), x_vec)
                repo2sample[repo]['validation'].append(ID)
            else:
                print_and_log(logger, "time_sample_outputevent not in "
                              "training or validation period!")
                pdb.set_trace()

    # with open(os.path.join(config['root_dir'], "data",
    #                        'each_repo_chain_length.pickle'), 'wb') as f:
    #     pickle.dump(each_repo_chain_length, f)
    # with open(os.path.join(config['root_dir'], "data",
    #                        'each_user_total_event_count.pickle'), 'wb') as f:
    #     pickle.dump(each_user_total_event_count, f)
    # with open(os.path.join(config['root_dir'], "data",
    #                        'each_event_type_total_count.pickle'), 'wb') as f:
    #     pickle.dump(each_event_type_total_count, f)

    # pdb.set_trace()

    print_and_log(logger, "number of chains used for "
                  "training and validation: {}".format(
                          len(unique_repo_train_vali)))

    print_and_log(logger, "could not find cluster for {} users.".format(
            len(user_has_no_cluster)))

    # et_pos, et_neg, uc_pos, uc_neg
    """
    number of events with that event type / the total number of events
    number of events without that event type / total number of events
    number of events with this user cluster / total number of events
    number of events without this user cluster / total number of events

    et_pos = []
    et_neg = []
    uc_pos = []
    uc_neg = []
    et_ids = sorted(list(id_2_eventtype.keys()))
    uc_ids = list(range(101))
    for i in et_ids:
        et_pos.append(0)
        et_neg.append(0)
    for i in uc_ids:
        uc_pos.append(0)
        uc_neg.append(0)
    total_events = 0
    for e in each_event_type_total_count:
        total_events += each_event_type_total_count[e]

    for i in et_ids:
        if i == 11:
            continue
        et_pos[i-1] = each_event_type_total_count[i]/total_events
        et_neg[i-1] = (total_events - each_event_type_total_count[
                i])/total_events
    et_neg[10] = float(1)

    for i in uc_ids:
        uc_pos[i] = each_cluster_total_event_count[i]/total_events
        uc_neg[i] = (total_events - each_cluster_total_event_count[
                i])/total_events

    with open(os.path.join(config['root_dir'], "data",
                           'et_pos.json'), 'w') as f:
        json.dump(et_pos, f)
    with open(os.path.join(config['root_dir'], "data",
                           'et_neg.json'), 'w') as f:
        json.dump(et_neg, f)
    with open(os.path.join(config['root_dir'], "data",
                           'uc_pos.json'), 'w') as f:
        json.dump(uc_pos, f)
    with open(os.path.join(config['root_dir'], "data",
                           'uc_neg.json'), 'w') as f:
        json.dump(uc_neg, f)

    pdb.set_trace()
    """

    if len(user_has_no_cluster) > 0:
        with open(os.path.join(config['exp_save_dir'], "dataset",
                               'user_has_no_cluster.pickle'), 'wb') as f:
            pickle.dump(user_has_no_cluster, f)

    with open(os.path.join(config['exp_save_dir'], "dataset",
                           'partition.json'), 'w') as f:
        json.dump(partition, f)

    with open(os.path.join(config['exp_save_dir'], "dataset",
                           'repo2sample.pickle'), 'wb') as f:
        pickle.dump(repo2sample, f)

    with open(os.path.join(config['exp_save_dir'], "dataset",
                           'unique_repo_train_vali.json'), 'w') as f:
        json.dump(unique_repo_train_vali, f)
    with open(os.path.join(config['exp_save_dir'], "dataset",
                           'unique_user_train_vali.json'), 'w') as f:
        json.dump(unique_user_train_vali, f)

    df = pd.DataFrame(labels)
    df.to_json(os.path.join(config['exp_save_dir'], "dataset", 'labels.json'))

    print_and_log(logger, "the number of training samples: {}".format(
            len(partition['train'])))
    print_and_log(logger, "the number of validation samples: {}".format(
            len(partition['validation'])))

    print_and_log(logger, "the number of unique repos in training and "
                  "validation samples: {}".format(
                          len(unique_repo_train_vali)))
    print_and_log(logger, "the number of unique users in training and "
                  "validation samples: {}".format(
                          len(unique_user_train_vali)))

    print_and_log(logger, "{} took {} min".format(
            "trainset valiset creation",
            (time.time()-create_dataset_start)/60))
    # pdb.set_trace()
    return (partition, labels)


class DataGenerator_ET_TD_UC_one_hot(keras.utils.Sequence):
    """
    Reference:
        https://stanford.edu/~shervine/blog/
            keras-how-to-generate-data-on-the-fly#data-generator
    """
    'Generates data for Keras'
    def __init__(self, data_path, list_IDs, labels,
                 batch_size, dim, window_size, et_classes, uc_classes,
                 shuffle=False):
        'Initialization'
        self.data_path = data_path
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.window_size = window_size
        self.et_classes = et_classes
        self.uc_classes = uc_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, window_size, dim)
        # Initialization
        X = np.empty((self.batch_size, self.window_size, self.dim))
        y_et = np.empty((self.batch_size))
        y_td = np.empty((self.batch_size))
        y_uc = np.empty((self.batch_size))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = np.load(os.path.join(
                    self.data_path, ID + '.npy'))

            # Store class
            y_et[i] = self.labels['event_type'][ID]
            y_td[i] = self.labels['time_delay'][ID]
            y_uc[i] = self.labels['user_cluster'][ID]

        y_et = keras.utils.to_categorical(y_et, num_classes=self.et_classes)
        y_uc = keras.utils.to_categorical(y_uc, num_classes=self.uc_classes)

        y = [y_et, y_td, y_uc]
        return X, y


class DataGenerator_ET_TD_UC_minmax(keras.utils.Sequence):
    """
    Reference:
        https://stanford.edu/~shervine/blog/
            keras-how-to-generate-data-on-the-fly#data-generator
    """
    'Generates data for Keras'
    def __init__(self, data_path, list_IDs, labels,
                 batch_size, dim, window_size, et_classes,
                 shuffle=False):
        'Initialization'
        self.data_path = data_path
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.window_size = window_size
        self.et_classes = et_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, window_size, dim)
        # Initialization
        X = np.empty((self.batch_size, self.window_size, self.dim))
        y_et = np.empty((self.batch_size))
        y_td = np.empty((self.batch_size))
        y_uc = np.empty((self.batch_size))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = np.load(os.path.join(
                    self.data_path, ID + '.npy'))

            # Store class
            y_et[i] = self.labels['event_type'][ID]
            y_td[i] = self.labels['time_delay'][ID]
            y_uc[i] = self.labels['user_cluster'][ID]

        y_et = keras.utils.to_categorical(y_et, num_classes=self.et_classes)

        y = [y_et, y_td, y_uc]
        return X, y
