#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:38:29 2019

@author: zhouhonglu
"""
from models.multitask_lstm.utils import categorical_focal_loss
from models.multitask_lstm.utils import set_logger
from models.multitask_lstm.utils import print_and_log
from models.multitask_lstm.utils import time_delay_normalization
from models.multitask_lstm.utils import load_jsongz_2_dataframe
from models.multitask_lstm.utils import insert_no_event_for_a_chain
from models.multitask_lstm.utils import get_time_delay
from models.multitask_lstm.utils import utc_timestamp
# from models.multitask_lstm.utils import (
#         preprocess_cascade_files_to_remove_unnesseary_chains)
# from models.multitask_lstm.utils import count_profile_repos

import os
import pdb
import time
from datetime import datetime as dt
import json
import pandas as pd
import numpy as np
import pickle

import keras
from keras import metrics
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout
from keras.losses import logcosh
from keras import losses
from keras.utils import multi_gpu_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
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
    lstm_one = LSTM(250, return_sequences=True, name="lstm_one")(
                    input_sequences)
    lstm_two = LSTM(150, name="lstm_two")(lstm_one)

    # branching event type
    dense_one_layer = Dense(128, activation='relu', name="dense_one")
    dense_one = dense_one_layer(lstm_two)

    dropout_one_layer = Dropout(0.5, name="dropout_one")
    dropout_one = dropout_one_layer(dense_one)

    dense_two_layer = Dense(64, activation='relu', name="dense_two")
    dense_two = dense_two_layer(dropout_one)

    dropout_two_layer = Dropout(0.5, name="dropout_two")
    dropout_two = dropout_two_layer(dense_two)

    event_type_output_layer = Dense(len(config['eventtype_2_id']),
                                    activation='softmax',
                                    name="event_type_output")
    event_type_output = event_type_output_layer(dropout_two)

    # branching time delay
    dense_four_layer = Dense(128, activation='relu', name="dense_four")
    dense_four = dense_four_layer(lstm_two)

    dropout_three_layer = Dropout(0.5, name="dropout_three")
    dropout_three = dropout_three_layer(dense_four)

    dense_five_layer = Dense(64, activation='relu', name="dense_five")
    dense_five = dense_five_layer(dropout_three)

    dropout_four_layer = Dropout(0.5, name="dropout_four")
    dropout_four = dropout_four_layer(dense_five)

    time_delay_output_layer = Dense(1, activation='linear',
                                    name="time_delay_output")
    time_delay_output = time_delay_output_layer(dropout_four)

    model = Model(inputs=input_sequences,
                  outputs=[event_type_output, time_delay_output])

    model = multi_gpu_model(model, gpus=config['num_gpu'])

    model.summary(print_fn=logger.info)

    print(model.summary())

    # get partition and labels
    with open(os.path.join(config['exp_save_dir'], "dataset",
                           'partition.json'), 'r') as f:
        partition = json.load(f)
    with open(os.path.join(config['exp_save_dir'], "dataset",
                           'labels.json'), 'r') as f:
        labels = json.load(f)

    # train val generator
    training_generator = DataGenerator_ET_TD(
            os.path.join(config['exp_save_dir'], "dataset"),
            partition['train'],
            labels, batch_size=config['batch_size'],
            dim=x_dim,
            window_size=config['window_size'],
            n_classes=len(config['eventtype_2_id']),
            shuffle=config['generator_shuffle'])
    validation_generator = DataGenerator_ET_TD(
            os.path.join(config['exp_save_dir'], "dataset"),
            partition['validation'],
            labels, batch_size=config['batch_size'],
            dim=x_dim,
            window_size=config['window_size'],
            n_classes=len(config['eventtype_2_id']),
            shuffle=config['generator_shuffle'])

    # callback
    if not os.path.exists(os.path.join(
            config['exp_save_dir'], "models", "vis")):
        os.makedirs(os.path.join(
                config['exp_save_dir'], "models", 'vis'))

    TensorBoard_callback = TensorBoard(
            log_dir=os.path.join(config['exp_save_dir'], "models", 'vis'),
            histogram_freq=0,
            write_graph=True,
            write_images=True,
            write_grads=False)
    """
    Keras TensorBoard Reference:
        https://keras.io/callbacks/#tensorboard

    launch: tensorboard --logdir=/full_path_to_your_logs
    """

    callbacks = [ModelCheckpoint(os.path.join(config['exp_save_dir'],
                                              "models",
                                              'model.hdf5'),
                                 monitor='val_loss',
                                 verbose=2),
                 TensorBoard_callback]

    # save train confg in case testing need it
    with open(os.path.join(config['exp_save_dir'], "models",
                           'train_config.pickle'), 'wb') as f:
        pickle.dump(config, f)
    print("{} saved!".format('train_config.pickle'))

    # model_history
    model_history = dict()

    # start training
    for epoch in range(config['num_epochs']):
        if epoch % (6+1) == 0:  # freeze time delay for 1 epoch
            print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!  "
                  "Epoch {}/{}    {}".format(
                          epoch+1, config['num_epochs'],
                          "---> train event type branch..."))

            dense_one_layer.trainable = True
            dropout_one_layer.trainable = True
            dense_two_layer.trainable = True
            dropout_two_layer.trainable = True
            event_type_output_layer.trainable = True

            dense_four_layer.trainable = False
            dropout_three_layer.trainable = False
            dense_five_layer.trainable = False
            dropout_four_layer.trainable = False
            time_delay_output_layer.trainable = False
        else:  # freeze event type for 6 epoch
            print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!  "
                  "Epoch {}/{}    {}".format(
                          epoch+1, config['num_epochs'],
                          "---> train time delay branch..."))
            dense_one_layer.trainable = False
            dropout_one_layer.trainable = False
            dense_two_layer.trainable = False
            dropout_two_layer.trainable = False
            event_type_output_layer.trainable = False

            dense_four_layer.trainable = True
            dropout_three_layer.trainable = True
            dense_five_layer.trainable = True
            dropout_four_layer.trainable = True
            time_delay_output_layer.trainable = True

        trainable_count = int(
            np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
        non_trainable_count = int(
            np.sum([K.count_params(p) for p in set(
                    model.non_trainable_weights)]))

        print('Total params: {:,}'.format(
                trainable_count + non_trainable_count))
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))

        model.compile(optimizer='adam',
                      loss=[categorical_focal_loss(
                              gamma=config['focalloss_gamma'],
                              alpha=config['focalloss_alpha']),
                            logcosh],
                      loss_weights=config['loss_weights'],
                      metrics={'event_type_output': (
                                       metrics.categorical_accuracy),
                               'time_delay_output': (
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
    if config['use_user_profile_features']:
        with open(config['user_profile_feat_path'], 'r') as f:
            user_profile_feat = json.load(f)
    if config['use_user_activity_features']:
        with open(config['user_activity_feat_path'], 'r') as f:
            user_act_feat = json.load(f)

    print_and_log(logger, "x_dim: {}".format(x_dim))

    partition = dict()
    partition['train'] = []
    partition['validation'] = []
    labels = dict()
    labels['event_type'] = dict()
    labels['time_delay'] = dict()

    cascades_path = os.path.join(config['cascade_dir'], "training", "github")

    # count_profile_repos(config, cascades_path, github_event_repos_set)
    # pdb.set_trace()

    # preprocess_cascade_files_to_remove_unnesseary_chains(
    #     config, cascades_path, github_event_repos_set)
    # pdb.set_trace()

    # load foldername2infoid or create it
    # Note to Hareesh: actually should be created with cascades!
    if not os.path.exists(os.path.join(
            cascades_path, "foldername2infoid.json")):
        with open(os.path.join(
                cascades_path, "infoid2foldername.json"), 'r') as f:
            infoid2foldername = json.load(f)
        foldername2infoid = dict()
        for infoid in infoid2foldername:
            foldername2infoid[infoid2foldername[infoid]] = infoid
        with open(os.path.join(
                cascades_path, "foldername2infoid.json"), 'w') as f:
            json.dump(foldername2infoid, f)
    else:
        with open(os.path.join(
                cascades_path, "foldername2infoid.json"), 'r') as f:
            foldername2infoid = json.load(f)

    all_id_list = [f for f in os.listdir(cascades_path) if "id_" in f]
    print_and_log(logger, "there are {} info_ids provided "
                          "for training.".format(len(all_id_list)))
    sample_id = 0
    chain_count = 0

    unique_repo_train = {}
    unique_user_train = {}

    for info_idx in range(len(all_id_list)):
        info_id = all_id_list[info_idx]
        print('processing {}, {}/{}'.format(
                round(info_idx/len(all_id_list), 2),
                info_idx,
                len(all_id_list)),
              end='\r')

        reoo_list = [f[:-8] for f in os.listdir(os.path.join(
                cascades_path, info_id)) if '.json.gz' in f]

        for repo in reoo_list:
            if (repo in config['repos_to_ignore']) or (
                    repo in github_event_repos_set):
                # it is a event repo or repo should be ignore
                print("it is a event repo or repo should be ignore, "
                      "shouldn't happen!")
                pdb.set_trace()
            chain_count += 1

            one_chain_pd = load_jsongz_2_dataframe(
                    os.path.join(cascades_path, info_id, repo + '.json.gz'))
            one_chain_pd = one_chain_pd.sort_values(by=['nodeTime'])

            # get the unique repos users in the training cascades
            unique_repo_train[repo] = []
            for user in one_chain_pd['nodeUserID']:
                unique_user_train[user] = []

            """
            # get the unique repos users in the training cascades
            # Note: this way more accurate but will increase running time a lot
            for user_idx in range(len(one_chain_pd['nodeUserID'])):
                user = one_chain_pd['nodeUserID'][user_idx]
                if (one_chain_pd['nodeTime'][user_idx] >= train_start) and (
                        one_chain_pd['nodeTime'][user_idx] <= vali_end+1):
                    unique_user_train[user] = []
                    unique_repo_train[repo] = []
            """

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
                insert_no_event_for_a_chain(config,
                                            one_chain_event, one_chain_time,
                                            one_chain_user)
            # if one_chain_event_new != one_chain_event:
            #     pdb.set_trace()

            one_chain_event = one_chain_event_new
            one_chain_time = one_chain_time_new
            one_chain_user = one_chain_user_new

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
                        time_sample_outputevent <= vali_end+1)):
                    continue

                input_event_type = \
                    one_chain_event[i-config['window_size']:i]
                input_time_delay = \
                    one_chain_time_delay[i-config['window_size']:i]
                input_user = one_chain_user[i-config['window_size']:i]

                output_event_type = \
                    one_chain_event[i]
                output_time_delay = \
                    one_chain_time_delay[i]
                # output_user = one_chain_user[i]

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

                    x_vec.append(x_j)

                if (time_sample_outputevent >= train_start) and (
                        time_sample_outputevent <= train_end+1):
                    partition['train'].append(ID)
                    labels['event_type'][ID] = output_event_type-1
                    labels['time_delay'][ID] = output_time_delay
                    np.save(os.path.join(config['exp_save_dir'], "dataset",
                                         ID + '.npy'), x_vec)

                elif (time_sample_outputevent >= vali_start) and (
                        time_sample_outputevent <= vali_end+1):
                    partition['validation'].append(ID)
                    labels['event_type'][ID] = output_event_type-1
                    labels['time_delay'][ID] = output_time_delay
                    np.save(os.path.join(config['exp_save_dir'], "dataset",
                                         ID + '.npy'), x_vec)
                else:
                    print_and_log(logger, "time_sample_outputevent not in "
                                  "training or validation period!")
                    pdb.set_trace()

    print_and_log(logger, "number of chains used for "
                  "training and validation: {}".format(chain_count))

    # pdb.set_trace()

    with open(os.path.join(config['exp_save_dir'], "dataset",
                           'partition.json'), 'w') as f:
        json.dump(partition, f)

    with open(os.path.join(config['exp_save_dir'], "dataset",
                           'unique_repo_train.json'), 'w') as f:
        json.dump(unique_repo_train, f)
    with open(os.path.join(config['exp_save_dir'], "dataset",
                           'unique_user_train.json'), 'w') as f:
        json.dump(unique_user_train, f)

    df = pd.DataFrame(labels)
    df.to_json(os.path.join(config['exp_save_dir'], "dataset", 'labels.json'))

    print_and_log(logger, "the number of training samples: {}".format(
            len(partition['train'])))
    print_and_log(logger, "the number of validation samples: {}".format(
            len(partition['validation'])))

    print_and_log(logger, "the number of unique repos in  training and "
                  "validation samples: {}".format(len(unique_repo_train)))
    print_and_log(logger, "the number of unique repos in  training and "
                  "validation samples: {}".format(len(unique_user_train)))

    print_and_log(logger, "{} took {} min".format(
            "trainset valiset creation",
            (time.time()-create_dataset_start)/60))
    # pdb.set_trace()
    return (partition, labels)


class DataGenerator_ET_TD(keras.utils.Sequence):
    """
    Reference:
        https://stanford.edu/~shervine/blog/
            keras-how-to-generate-data-on-the-fly#data-generator
    """
    'Generates data for Keras'
    def __init__(self, data_path, list_IDs, labels,
                 batch_size, dim, window_size, n_classes,
                 shuffle=False):
        'Initialization'
        self.data_path = data_path
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.window_size = window_size
        self.n_classes = n_classes
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
        y_et = np.empty((self.batch_size), dtype=int)
        y_td = np.empty((self.batch_size), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = np.load(os.path.join(
                    self.data_path, ID + '.npy'))

            # Store class
            y_et[i] = self.labels['event_type'][ID]
            y_td[i] = self.labels['time_delay'][ID]

        if self.n_classes > 1:
            y_et = keras.utils.to_categorical(y_et, num_classes=self.n_classes)

        y = [y_et, y_td]
        return X, y
