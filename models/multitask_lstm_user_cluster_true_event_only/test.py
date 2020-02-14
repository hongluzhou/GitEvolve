#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:39:11 2019

@author: zhouhonglu
"""
from models.multitask_lstm_user_cluster_true_event_only.utils import (
        categorical_focal_loss)
from models.multitask_lstm_user_cluster_true_event_only.utils import (
        set_logger)
from models.multitask_lstm_user_cluster_true_event_only.utils import (
        print_and_log)
from models.multitask_lstm_user_cluster_true_event_only.utils import (
        time_delay_normalization)
from models.multitask_lstm_user_cluster_true_event_only.utils import (
        time_delay_normalization_reverse)
from models.multitask_lstm_user_cluster_true_event_only.utils import (
        load_jsongz_2_dataframe)
# from models.multitask_lstm_user_cluster_true_event_only.utils import (
#         insert_no_event_for_a_chain_new)
# from models.multitask_lstm_user_cluster_true_event_only.utils import (
#         insert_no_event_for_a_sim_GTchain)
from models.multitask_lstm_user_cluster_true_event_only.utils import (
        get_time_delay)
from models.multitask_lstm_user_cluster_true_event_only.utils import (
        utc_timestamp)
# from models.multitask_lstm_user_cluster_true_event_only.utils import (
#         insert_no_event_for_a_GTchain_who_has_no_event_at_all)

import os
import pdb
import time
from datetime import datetime as dt
import dill
import json
import pandas as pd
import numpy as np
import pickle
from fastdtw import fastdtw

import keras
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

pd.set_option('display.max_columns', None)


def test(config, process_create_dataset=True):
    if process_create_dataset:
        # create testset on the fly

        if config['given_gt']:
            testset = testset_creation_given_gt(config)

        else:
            testset = testset_creation(config)
    else:
        # load testset

        if config['given_gt']:
            testset_save_path = os.path.join(
                    config['exp_save_dir'], "dataset",
                    'testset_given_gt.pickle')
            with open(testset_save_path, 'rb') as handle:
                testset = pickle.load(handle)
            print("testset_given_gt.pickle loaded!")

        else:
            testset_save_path = os.path.join(
                    config['exp_save_dir'], "dataset",
                    'testset.pickle')
            with open(testset_save_path, 'rb') as handle:
                testset = pickle.load(handle)
            print("testset.pickle loaded!")

    # load model
    custom_object = {
            'categorical_focal_loss_fixed': dill.loads(dill.dumps(
                    categorical_focal_loss(
                            gamma=config['focalloss_gamma'],
                            alpha=config['focalloss_alpha']))),
            'categorical_focal_loss': categorical_focal_loss}

    trained_model_path = os.path.join(
            config['exp_save_dir'], "models",
            'model-{}.hdf5'.format(config['load_model_epoch']))

    if not os.path.exists(trained_model_path):
        raise FileNotFoundError(
            'Model file `{}` does not exist.'.format(trained_model_path))

    if config['et_loss'] == 'focalloss':
        model = keras.models.load_model(trained_model_path,
                                        custom_objects=custom_object)
    else:
        model = keras.models.load_model(trained_model_path)

    # simulation

    if config['given_gt']:
        simulation_given_gt(config, model, testset)
    else:
        simulation(config, model, testset)

    return


def simulation(config, model, testset):
    """
    user feat not used!
    """
    sim_took_time = time.time()

    # preparation
    result_save_path = os.path.join(config['exp_save_dir'],
                                    'test_result-epoch{}'.format(
                                            config['load_model_epoch']))
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    keep_pred_max = config['keep_pred_max']
    print("keep prediction threshold: {}".format(keep_pred_max))

    eventtype_2_id = config['eventtype_2_id']
    id_2_eventtype = dict(zip(eventtype_2_id.values(),
                              eventtype_2_id.keys()))

    test_end_utc = utc_timestamp(config['sim_period']['end'])

    # get x_dim
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
                            '10log10_xplus1'))
            this_repo_profile_feat += repo_profile_raw[repo]['user_type']
            repo_profile_feat[repo] = this_repo_profile_feat
    if config['use_repo_idx_features']:
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
        print("max_sum_act_feat: {}".format(
                max_sum_act_feat))
        print("max_sum_act_feat_vec: {}".format(
                max_sum_act_feat_vec))
    if config['use_user_profile_features']:
        pass
    if config['use_user_activity_features']:
        pass
    if config['use_user_cluster_one_hot']:
        with open(config['user_cluster_path'], 'r') as f:
            user_clusters = json.load(f)
        padding_event_user_cluster = max(user_clusters.values()) + 1
        user_clusters[config['empty_user']] = padding_event_user_cluster
    if config['use_user_cluster_minmax']:
        with open(config['user_cluster_path'], 'r') as f:
            user_clusters = json.load(f)
        max_cluster_id = max(user_clusters.values())
        min_cluster_id = min(user_clusters.values())
        padding_event_user_cluster = max_cluster_id + 1
        max_cluster_id = padding_event_user_cluster
        # max_minus_min = max_cluster_id - min_cluster_id
        if min_cluster_id != 0:
            print("min cluster id is not 0! Need to examine code!")
            pdb.set_trace()
        user_clusters[config['empty_user']] = padding_event_user_cluster
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
        cluster_profile_feat[padding_event_user_cluster] = [0] * config[
                'dim_cluster_profile_features']
        print("max_sum_profile_feat: {}".format(
                max_sum_profile_feat))
        print("max_sum_profile_feat_vec: {}".format(
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
        print("max_sum_act_feat: {}".format(
                max_sum_act_feat))
        print("max_sum_act_feat_vec: {}".format(
                max_sum_act_feat_vec))

    print("x_dim: {}".format(x_dim))

    # save test confg in case testing need it
    with open(os.path.join(result_save_path,
                           'test_config.pickle'), 'wb') as f:
        pickle.dump(config, f)
    print("{} saved!".format('test_config.pickle'))

    # pdb.set_trace()

    # get predictions
    print("==========================================")
    print("start testing...")
    pred_all_event_id = []
    pred_all_event_type = []
    pred_all_time_delay = []
    pred_all_user_cluster = []
    chains_applied_keep_pred = []

    # for each repo-info in the testset
    for i in range(len(testset['X_test'])):
        repo = testset['repo'][i]
        print("testing {}/{}   repo: {}...".format(
                i+1, len(testset['X_test']), repo))

        keep_pred = dict()
        keep_pred_cond_pass = True
        applied = False

        # get the largest time delay this chain could have
        largest_time_delay_hours = \
            (test_end_utc - testset['input_last_event_time'][i])/3600
        test_time_delay_total = 0

        # start to simulate this chain
        pred_event_id = []
        pred_event_type = []
        pred_time_delay = []
        pred_user_cluster = []

        # predict the first event
        sample_input_this_chain = np.zeros((
                1, config['window_size'], x_dim))
        sample_next_this_chain = np.zeros((
                1, config['window_size'], x_dim))

        sample_input_this_chain[0, :] = testset['X_test'][i]
        sample_next_this_chain[0, :config['window_size']-1, :] = \
            sample_input_this_chain[0, 1:, :]

        if config['use_repo_embedding']:
            embed_vec = embed_vec_all[repo[:22]+'/'+repo[23:45]]

        (y_pred, time_delay, user_cluster) = model.predict(
                sample_input_this_chain, batch_size=1, verbose=0)

        # change one hot encoding output event into actual event id
        predict_event = np.argmax(y_pred[0]) + 1

        # modify time_delay and time_delay_hour
        time_delay = time_delay[0][0]
        if config['time_delay_normalization_func'] is not None:
            time_delay_hour = time_delay_normalization_reverse(
                    time_delay,
                    config['time_delay_normalization_func'])
            keep_pred[round(time_delay_hour,
                            config['keep_pred_round'])] = 1
        else:
            print("not implemented!")
            pdb.set_trace()

        if config['use_user_cluster_one_hot'] and (
                not config['use_user_cluster_minmax']):
            predict_user_cluster = np.argmax(user_cluster[0])

        else:
            # predict_user_cluster = round(user_cluster * max_minus_min)
            predict_user_cluster = int(round(user_cluster[0][0]))

        # to make sure we can find profile feature for all predicted cluster
        if predict_user_cluster > padding_event_user_cluster:
            predict_user_cluster = padding_event_user_cluster
        if predict_user_cluster < 0:
            predict_user_cluster = 0

        # pdb.set_trace()
        if (test_time_delay_total + time_delay_hour <=
                largest_time_delay_hours) and (keep_pred_cond_pass):
            # add predicted new event
            pred_event_id.append(predict_event)
            pred_event_type.append(id_2_eventtype[predict_event])
            pred_time_delay.append(time_delay_hour)
            pred_user_cluster.append(predict_user_cluster)
            test_time_delay_total += time_delay_hour

            # feature calculation for predicted new event
            dim_pointer = 0

            if config['use_repo_embedding']:
                sample_next_this_chain[0, -1, :dim_pointer + config[
                        'dim_repo_embedding']] = embed_vec
                dim_pointer += config['dim_repo_embedding']

            if config['use_repo_profile_features']:
                try:
                    sample_next_this_chain[
                            0, -1, dim_pointer:dim_pointer + config[
                                    'dim_repo_profile_features']] = \
                                repo_profile_feat[repo]
                except KeyError:
                    sample_next_this_chain[
                            0, -1, dim_pointer:dim_pointer + config[
                                    'dim_repo_profile_features']] = \
                                [0] * config['dim_repo_profile_features']
                dim_pointer += config['dim_repo_profile_features']

            if config['use_repo_idx_features']:
                try:
                    sample_next_this_chain[
                            0, -1, dim_pointer:dim_pointer + config[
                                    'dim_repo_idx_features']] = \
                                load_repo_train_vali[repo]
                except KeyError:
                    sample_next_this_chain[
                            0, -1, dim_pointer:dim_pointer + config[
                                    'dim_repo_idx_features']] = [0]
                dim_pointer += config['dim_repo_idx_features']

            if config['use_repo_activity_features']:
                if predict_user_cluster == padding_event_user_cluster:
                    sample_next_this_chain[
                        0, -1, dim_pointer: dim_pointer + config[
                                'dim_repo_activity_features']
                        ] = [0] * config['dim_repo_activity_features']
                else:
                    try:
                        repo_thiscluster_act_feat = repo_act_feat[
                                repo][predict_user_cluster]
                        repo_allcluster_act_feat = repo_act_feat[
                                repo]['all_cluster']

                        sample_next_this_chain[
                            0, -1, dim_pointer: dim_pointer + config[
                                    'dim_repo_activity_features']
                            ] = (repo_thiscluster_act_feat +
                                 repo_allcluster_act_feat)
                    except KeyError:
                        sample_next_this_chain[
                            0, -1, dim_pointer: dim_pointer + config[
                                    'dim_repo_activity_features']
                            ] = [0] * config['dim_repo_activity_features']
                dim_pointer += config['dim_repo_activity_features']

            if config['use_user_profile_features']:
                pass

            if config['use_user_activity_features']:
                pass

            if config['use_event_type_one_hot']:
                sample_next_this_chain[0, -1, dim_pointer:dim_pointer + len(
                                    config['eventtype_2_id'])] = \
                    y_pred[0]
                dim_pointer += len(config['eventtype_2_id'])

            if config['use_time_delay_features']:
                sample_next_this_chain[
                        0, -1, dim_pointer: dim_pointer + config[
                                'dim_time_delay_features']] = time_delay
                dim_pointer += config['dim_time_delay_features']

            if config['use_user_cluster_one_hot']:
                sample_next_this_chain[
                        0, -1, dim_pointer: dim_pointer + config[
                                'dim_user_cluster_one_hot']
                        ] = user_cluster[0]
                dim_pointer += config['dim_user_cluster_one_hot']

            if config['use_user_cluster_minmax']:
                sample_next_this_chain[
                        0, -1, dim_pointer: dim_pointer + config[
                                'dim_user_cluster_minmax']
                        ] = user_cluster[0]
                dim_pointer += config['dim_user_cluster_minmax']

            if config['use_cluster_profile_features']:
                this_cluster_profile_feat = cluster_profile_feat[
                        predict_user_cluster]
                sample_next_this_chain[
                        0, -1, dim_pointer: dim_pointer + config[
                                'dim_cluster_profile_features']
                        ] = this_cluster_profile_feat
                dim_pointer += config['dim_cluster_profile_features']

            if config['use_cluster_activity_features']:
                if predict_user_cluster == padding_event_user_cluster:
                    sample_next_this_chain[
                        0, -1, dim_pointer: dim_pointer + config[
                                'dim_cluster_activity_features']
                        ] = [0] * config['dim_cluster_activity_features']
                else:
                    try:
                        cluster_thisrepo_act_feat = cluster_act_feat[
                                    predict_user_cluster][repo]
                        cluster_allrepo_act_feat = cluster_act_feat[
                                    predict_user_cluster]['all_repo']

                        sample_next_this_chain[
                            0, -1, dim_pointer: dim_pointer + config[
                                    'dim_cluster_activity_features']
                            ] = (cluster_thisrepo_act_feat +
                                 cluster_allrepo_act_feat)
                    except KeyError:
                        sample_next_this_chain[
                            0, -1, dim_pointer: dim_pointer + config[
                                    'dim_cluster_activity_features']
                            ] = [0] * config[
                                    'dim_cluster_activity_features']
                dim_pointer += config['dim_cluster_activity_features']

            if dim_pointer != x_dim:
                print("dim_pointer != x_dim")
                pdb.set_trace()

            # print("start testing, {}/{}, {}, {}, {}, {}".format(
            #         i, len(testset['X_test']),
            #         id_2_eventtype[predict_event],
            #         time_delay_hour, test_time_delay_total,
            #         largest_time_delay_hours))
            # pdb.set_trace()
            while ((test_time_delay_total +
                    time_delay_hour) <= largest_time_delay_hours) and (
                    keep_pred_cond_pass):
                (y_pred, time_delay, user_cluster) = model.predict(
                    sample_next_this_chain, batch_size=1, verbose=0)

                # change one hot encoding output event into actual event id
                predict_event = np.argmax(y_pred[0]) + 1

                # modify time_delay, time_delay_hour or keep_pred_cond_pass
                time_delay = time_delay[0][0]
                if config['time_delay_normalization_func'] is not None:
                    time_delay_hour = time_delay_normalization_reverse(
                            time_delay,
                            config['time_delay_normalization_func'])
                else:
                    print("not implemented!")
                    pdb.set_trace()
                if round(time_delay_hour,
                         config['keep_pred_round']) in keep_pred:
                    keep_pred[round(time_delay_hour,
                                    config['keep_pred_round'])] += 1
                else:
                    keep_pred = dict()
                    keep_pred[round(time_delay_hour,
                                    config['keep_pred_round'])] = 1
                if keep_pred[round(time_delay_hour,
                                   config['keep_pred_round']
                                   )] <= keep_pred_max:
                    keep_pred_cond_pass = True
                else:
                    keep_pred_cond_pass = False
                    applied = True

                if config['use_user_cluster_one_hot'] and (
                        not config['use_user_cluster_minmax']):
                    predict_user_cluster = np.argmax(user_cluster[0])

                else:
                    # predict_user_cluster = round(
                    #         user_cluster * max_minus_min)
                    predict_user_cluster = int(round(user_cluster[0][0]))

                # make sure we can find profile feat for predicted cluster
                if predict_user_cluster > padding_event_user_cluster:
                    predict_user_cluster = padding_event_user_cluster
                if predict_user_cluster < 0:
                    predict_user_cluster = 0

                # update sample_next_this_chain
                sample_next_this_chain_new = np.zeros((
                        1, config['window_size'], x_dim))
                sample_next_this_chain_new[0, :config['window_size']-1, :] = \
                    sample_next_this_chain[0, 1:, :]
                sample_next_this_chain = sample_next_this_chain_new

                if ((test_time_delay_total +
                     time_delay_hour) <= largest_time_delay_hours) and (
                        keep_pred_cond_pass):
                    # add predicted new event
                    pred_event_id.append(predict_event)
                    pred_event_type.append(
                            id_2_eventtype[predict_event])
                    pred_time_delay.append(time_delay_hour)
                    pred_user_cluster.append(predict_user_cluster)
                    test_time_delay_total += time_delay_hour

                    # feature calculation for predicted new event node
                    dim_pointer = 0
                    if config['use_repo_embedding']:
                        sample_next_this_chain[0, -1, :dim_pointer + config[
                                'dim_repo_embedding']] = embed_vec
                        dim_pointer += config['dim_repo_embedding']

                    if config['use_repo_profile_features']:
                        try:
                            sample_next_this_chain[
                                    0, -1, dim_pointer:dim_pointer + config[
                                            'dim_repo_profile_features']
                                    ] = repo_profile_feat[repo]
                        except KeyError:
                            sample_next_this_chain[
                                    0, -1, dim_pointer:dim_pointer + config[
                                            'dim_repo_profile_features']
                                    ] = [0] * config[
                                            'dim_repo_profile_features']
                        dim_pointer += config['dim_repo_profile_features']

                    if config['use_repo_idx_features']:
                        try:
                            sample_next_this_chain[
                                    0, -1, dim_pointer:dim_pointer + config[
                                            'dim_repo_idx_features']] = \
                                        load_repo_train_vali[repo]
                        except KeyError:
                            sample_next_this_chain[
                                    0, -1, dim_pointer:dim_pointer + config[
                                            'dim_repo_idx_features']] = [0]
                        dim_pointer += config['dim_repo_idx_features']

                    if config['use_repo_activity_features']:
                        if (predict_user_cluster ==
                                padding_event_user_cluster):
                            sample_next_this_chain[
                                0, -1, dim_pointer: dim_pointer + config[
                                        'dim_repo_activity_features']
                                ] = [0] * config[
                                        'dim_repo_activity_features']
                        else:
                            try:
                                repo_thiscluster_act_feat = repo_act_feat[
                                        repo][predict_user_cluster]
                                repo_allcluster_act_feat = repo_act_feat[
                                        repo]['all_cluster']

                                sample_next_this_chain[
                                    0, -1, dim_pointer: dim_pointer + config[
                                            'dim_repo_activity_features']
                                    ] = (repo_thiscluster_act_feat +
                                         repo_allcluster_act_feat)
                            except KeyError:
                                sample_next_this_chain[
                                    0, -1, dim_pointer: dim_pointer + config[
                                            'dim_repo_activity_features']
                                    ] = [0] * config[
                                            'dim_repo_activity_features']
                        dim_pointer += config['dim_repo_activity_features']

                    if config['use_user_profile_features']:
                        pass

                    if config['use_user_activity_features']:
                        pass

                    if config['use_event_type_one_hot']:
                        sample_next_this_chain[
                                0, -1, dim_pointer:dim_pointer + len(
                                        config['eventtype_2_id'])] = \
                            y_pred[0]
                        dim_pointer += len(config['eventtype_2_id'])

                    if config['use_time_delay_features']:
                        sample_next_this_chain[
                                0, -1, dim_pointer: dim_pointer + config[
                                        'dim_time_delay_features']] = \
                            time_delay
                        dim_pointer += config['dim_time_delay_features']

                    if config['use_user_cluster_one_hot']:
                        sample_next_this_chain[
                                0, -1, dim_pointer: dim_pointer + config[
                                        'dim_user_cluster_one_hot']
                                ] = user_cluster[0]
                        dim_pointer += config['dim_user_cluster_one_hot']

                    if config['use_user_cluster_minmax']:
                        sample_next_this_chain[
                                0, -1, dim_pointer: dim_pointer + config[
                                        'dim_user_cluster_minmax']
                                ] = user_cluster[0]
                        dim_pointer += config['dim_user_cluster_minmax']

                    if config['use_cluster_profile_features']:
                        this_cluster_profile_feat = cluster_profile_feat[
                                predict_user_cluster]
                        sample_next_this_chain[
                                0, -1, dim_pointer: dim_pointer + config[
                                        'dim_cluster_profile_features']
                                ] = this_cluster_profile_feat
                        dim_pointer += config['dim_cluster_profile_features']

                    if config['use_cluster_activity_features']:
                        if (predict_user_cluster ==
                                padding_event_user_cluster):
                            sample_next_this_chain[
                                0, -1, dim_pointer: dim_pointer + config[
                                        'dim_cluster_activity_features']
                                ] = [0] * config[
                                        'dim_cluster_activity_features']
                        else:
                            try:
                                cluster_thisrepo_act_feat = cluster_act_feat[
                                            predict_user_cluster][repo]
                                cluster_allrepo_act_feat = cluster_act_feat[
                                            predict_user_cluster]['all_repo']

                                sample_next_this_chain[
                                    0, -1, dim_pointer: dim_pointer + config[
                                            'dim_cluster_activity_features']
                                    ] = (cluster_thisrepo_act_feat +
                                         cluster_allrepo_act_feat)
                            except KeyError:
                                sample_next_this_chain[
                                    0, -1, dim_pointer: dim_pointer + config[
                                            'dim_cluster_activity_features']
                                    ] = [0] * config[
                                            'dim_cluster_activity_features']
                        dim_pointer += config[
                                'dim_cluster_activity_features']

                    if dim_pointer != x_dim:
                        print("dim_pointer != x_dim")
                        pdb.set_trace()

                    # print(repo)
                    # print(sample_next_this_chain)
                    # print(time_delay, time_delay_hour, predict_event)
                    # pdb.set_trace()

                    # print("start testing, {}/{}, {}, {}, {}, {}, "
                    #       "{}, {}".format(
                    #         i, len(testset['X_test']),
                    #         id_2_eventtype[predict_event],
                    #         round(time_delay_hour, 2),
                    #         round(test_time_delay_total, 2),
                    #         round(largest_time_delay_hours, 2),
                    #         len(pred_event_id),
                    #         len(testset['gt_time_delay'][i])))

        pred_all_event_id.append(pred_event_id)
        pred_all_event_type.append(pred_event_type)
        pred_all_time_delay.append(pred_time_delay)
        pred_all_user_cluster.append(pred_user_cluster)
        if applied:
            chains_applied_keep_pred.append(repo)
        print("Testing Result:\n"
              "Repo chain: {}\n"
              "ground truth chain length: {}\n"
              "predicted chain length: {}\n"
              "time delay DTW: {}\n"
              "applied keep prediction threshold? {} (threshold {})\n\n"
              "actual events: {}\n\n"
              "predicted events: {}\n\n"
              "actual time delay: {}\n\n"
              "predicted time delay: {}\n\n"
              "actual user cluster: {}\n\n"
              "predicted user cluster: {}\n\n"
              "==========================================\n".format(
                      repo,
                      len(testset['gt_time_delay'][i]),
                      len(pred_event_id),
                      fastdtw(testset['gt_time_delay'][i],
                              pred_time_delay)[0],
                      applied,
                      keep_pred_max,
                      testset['gt_event_type'][i],
                      pred_event_type,
                      testset['gt_time_delay'][i],
                      pred_time_delay,
                      testset['gt_user_cluster'][i],
                      pred_user_cluster
                      ))
        # pdb.set_trace()

    result = dict()
    result['chain_name'] = testset['repo']
    result['pred_all_event_id'] = pred_all_event_id
    result['pred_all_event_type'] = pred_all_event_type
    result['pred_all_time_delay'] = pred_all_time_delay
    result['pred_all_user_cluster'] = pred_all_user_cluster
    result['gt_all_event_id'] = testset['gt_event_id']
    result['gt_all_event_type'] = testset['gt_event_type']
    result['gt_all_time_delay'] = testset['gt_time_delay']
    result['gt_all_user_cluster'] = testset['gt_user_cluster']
    result['chains_applied_keep_pred'] = chains_applied_keep_pred
    result['simulation_runningtime_inhours'] = (
            time.time() - sim_took_time)/3600

    with open(os.path.join(result_save_path,
                           'result.pickle'), 'wb') as handle:
        pickle.dump(result, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print('\n\nresult.pickle saved in {}'.format(result_save_path))
    print("simulation took {} hours".format(
            result['simulation_runningtime_inhours']))
    return


def testset_creation(config, GT_avail=True):
    create_dataset_start = time.time()

    logger = set_logger(os.path.join(config['exp_save_dir'],
                                     'testset_creation_' +
                                     dt.now().strftime("%Y-%m-%dT%H-%M-%SZ") +
                                     '.log'))
    print_and_log(logger, '{}'.format(
                'testset creation...'))

    testset = dict()
    testset['X_test'] = []
    testset['repo'] = []
    testset['input_last_event_time'] = []
    testset['gt_event_type'] = None
    testset['gt_event_id'] = None
    testset['gt_time_delay'] = None
    testset['gt_user'] = None
    testset['gt_user_cluster'] = None
    X_test = []

    eventtype_2_id = config['eventtype_2_id']
    id_2_eventtype = dict(zip(eventtype_2_id.values(),
                              eventtype_2_id.keys()))

    sim_start = utc_timestamp(config['sim_period']['start'])
    sim_end = utc_timestamp(config['sim_period']['end'])

    with open(os.path.join(config['exp_save_dir'], "dataset",
                           'unique_repo_train_vali.json'), 'r') as f:
        unique_repo_train_vali = json.load(f)
    repo_list = list(unique_repo_train_vali.keys())

    with open(config['github_event_repos_path'], 'rb') as f:
        github_event_repos = pickle.load(f)
    github_event_repos_set = set(
            [repo[:22]+'-'+repo[23:] for repo in github_event_repos])

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
                            '10log10_xplus1'))
            this_repo_profile_feat += repo_profile_raw[repo]['user_type']
            repo_profile_feat[repo] = this_repo_profile_feat
    if config['use_repo_idx_features']:
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
        padding_event_user_cluster = max(user_clusters.values()) + 1
        user_clusters[config['empty_user']] = padding_event_user_cluster
    if config['use_user_cluster_minmax']:
        with open(config['user_cluster_path'], 'r') as f:
            user_clusters = json.load(f)
        max_cluster_id = max(user_clusters.values())
        min_cluster_id = min(user_clusters.values())
        padding_event_user_cluster = max_cluster_id + 1
        max_cluster_id = padding_event_user_cluster
        max_minus_min = max_cluster_id - min_cluster_id
        if min_cluster_id != 0:
            print("min cluster id is not 0! Need to examine code!")
            pdb.set_trace()
        user_clusters[config['empty_user']] = padding_event_user_cluster
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
        cluster_profile_feat[padding_event_user_cluster] = [0] * config[
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

    unique_repo_test = dict()
    unique_user_test = dict()

    user_has_no_cluster = set()

    # gather simulation sample input
    print_and_log(logger, "gather simulation sample input...")
    for repo in repo_list:  # for each cascade chain
        if (repo in config['repos_to_ignore']) or (
                repo in github_event_repos_set):
            # it is a event repo or repo should be ignore
            continue

        one_chain_pd = load_jsongz_2_dataframe(
                os.path.join(config['cascade_dir'], repo + '.json.gz'))
        # get all events before sim
        one_chain_pd = one_chain_pd.loc[
                    one_chain_pd['nodeTime'] < sim_start]
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

#        (one_chain_event_new, one_chain_time_new,
#         one_chain_user_new) = \
#            insert_no_event_for_a_chain_new(config, one_chain_event,
#                                            one_chain_time, one_chain_user,
#                                            sim_start)
        # if one_chain_event_new != one_chain_event:
        #     pdb.set_trace()

#        one_chain_event = one_chain_event_new
#        one_chain_time = one_chain_time_new
#        one_chain_user = one_chain_user_new

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

        # input_last_event_time
        testset['input_last_event_time'].append(one_chain_time[-1])
        testset['repo'].append(repo)

        input_event_type = one_chain_event[-config['window_size']:]
        input_time_delay = one_chain_time_delay[-config['window_size']:]
        input_user = one_chain_user[i-config['window_size']:i]
        input_cluster = []
        for user in input_user:
            try:
                input_cluster.append(user_clusters[user])
            except KeyError:
                # user_has_no_cluster.add(user)
                # input_cluster.append(user_clusters['no_event_user'])
                pdb.set_trace()

        # initialize input vector, and output vector for this sample
        x_vec = []

        # load node embeding vector
        if config['use_repo_embedding']:
            chain_temp = repo[:22] + '/' + repo[23:]
            try:
                embed_vec = np.array(embed_vec_all[chain_temp])
            except KeyError:
                print_and_log(logger, "Could not find embedding vector "
                              "for {}!".format(chain_temp))
                pdb.set_trace()
                # embed_vec = np.random.rand(1, config["embed_vec_len"])

        # input feature vector
        if len(input_event_type) < config['window_size']:
            print("len(input_event_type) < config['window_size']")
            pdb.set_trace()

        for j in range(len(input_event_type)):  # for each event node
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
                if input_cluster[j] == padding_event_user_cluster:
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
                if input_user[j] == config['empty_user']:
                    x_j += [0] * config['dim_user_profile_features']
                else:
                    try:
                        x_j += user_profile_feat[input_user[j]]
                    except KeyError:
                        x_j += [0] * config[
                                'dim_user_profile_features']

            if config['use_user_activity_features']:
                if input_user[j] == config['empty_user']:
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
                    try:
                        allrepo_feat = \
                            user_act_feat[input_user[j]]['all']
                    except KeyError:
                        # this user does not have event in test in GT
                        allrepo_feat = \
                            [0] * int(config[
                                    'dim_user_activity_features']/2)
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
                if input_cluster[j] == padding_event_user_cluster:
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
                        x_j += [0] * config[
                            'dim_cluster_activity_features']

            x_vec.append(x_j)

        X_test.append(x_vec)

    testset['X_test'] = np.array(X_test)

    print_and_log(logger, "X_test.shape: {}".format(
            testset['X_test'].shape))

    # gather simulation sample output
    if GT_avail:
        print_and_log(logger, "ground truth available. \n"
                      "gather simulation sample output...")
        testset['gt_event_type'] = []
        testset['gt_time_delay'] = []
        testset['gt_event_id'] = []
        testset['gt_user'] = []
        testset['gt_user_cluster'] = []

        # for each gathered simulation sample input
        for i in range(len(testset['repo'])):
            repo_this_sample = testset['repo'][i]

            one_chain_pd = load_jsongz_2_dataframe(
                    os.path.join(config['cascade_dir'],
                                 repo_this_sample+'.json.gz'))
            # get events during sim
            one_chain_pd = one_chain_pd.loc[(
                    one_chain_pd['nodeTime'] >= sim_start) & (
                    one_chain_pd['nodeTime'] <= sim_end)]
            one_chain_pd = one_chain_pd.sort_values(by=['nodeTime'])

            one_chain_event_id = []
            one_chain_time = []
            one_chain_user = []

            unique_repo_test[repo_this_sample] = []

            if len(one_chain_pd) == 0:
                # this repo has no events in testing period
                # need to insert no event till end of testing
                input_last_event_time_this_chain = testset[
                            'input_last_event_time'][i]
#                (one_chain_event_new,
#                 one_chain_time_new,
#                 one_chain_user_new
#                 ) = insert_no_event_for_a_GTchain_who_has_no_event_at_all(
#                         config, one_chain_event_id, one_chain_time,
#                         one_chain_user, input_last_event_time_this_chain,
#                         sim_end)
            else:
                # this sample chain has events in testing period
                for user in one_chain_pd['nodeUserID']:
                    unique_user_test[user] = []

                one_chain_event_id += [eventtype_2_id[
                        event] for event in one_chain_pd[
                        'actionType']]
                one_chain_time += [time for time in one_chain_pd[
                        'nodeTime']]
                one_chain_user += [user for user in one_chain_pd[
                        'nodeUserID']]
                input_last_event_time_this_chain = testset[
                        'input_last_event_time'][i]

                if min(one_chain_time) < sim_start:
                    print("min(one_chain_time) < sim_start")
                    pdb.set_trace()

#                (one_chain_event_new, one_chain_time_new,
#                 one_chain_user_new) = \
#                    insert_no_event_for_a_sim_GTchain(
#                            config,
#                            one_chain_event_id, one_chain_time,
#                            one_chain_user,
#                            input_last_event_time_this_chain,
#                            sim_end
#                            )

            # if one_chain_event_new != one_chain_event:
            #     pdb.set_trace()

#            one_chain_event_id = one_chain_event_new
#            one_chain_time = one_chain_time_new
#            one_chain_user = one_chain_user_new

            one_chain_event_type = [id_2_eventtype[
                    event] for event in one_chain_event_id]
            one_chain_user_cluster = []

            # no mather one-hot or normalized, here need true cluster
            for user in one_chain_user:
                try:
                    one_chain_user_cluster.append(
                            user_clusters[user])
                except KeyError:
                    user_has_no_cluster.add(user)
                    one_chain_user_cluster.append(
                            padding_event_user_cluster)

            # calculate time delay sequence
            one_chain_time_delay = []

            if len(one_chain_time) > 0:
                time_delay = get_time_delay(
                        input_last_event_time_this_chain,
                        one_chain_time[0],
                        'float')[1]
                # NO NEED TO DO normalization FOR GT !!!
                # if config['time_delay_normalization_func'] is not None:
                #     time_delay = time_delay_normalization(
                #             time_delay,
                #             config['time_delay_normalization_func'])
                one_chain_time_delay.append(time_delay)
                for j in range(1, len(one_chain_time)):
                    time_delay = get_time_delay(one_chain_time[j-1],
                                                one_chain_time[j],
                                                'float')[1]
                    # NO NEED TO DO normalization FOR GT !!!
                    # if config[
                    #       'time_delay_normalization_func'] is not None:
                    #     time_delay = time_delay_normalization(
                    #             time_delay,
                    #             config['time_delay_normalization_func'])
                    one_chain_time_delay.append(time_delay)

            testset['gt_event_id'].append(one_chain_event_id)
            testset['gt_event_type'].append(one_chain_event_type)
            testset['gt_time_delay'].append(one_chain_time_delay)
            testset['gt_user'].append(one_chain_user)
            testset['gt_user_cluster'].append(one_chain_user_cluster)

            # pdb.set_trace()

    print_and_log(logger, "could not find cluster for {} users.".format(
            len(user_has_no_cluster)))

#    # remove repo has no events in testing period since next event gt unkown
#    # because we don't have no event in one month
#    testset_new = dict()
#    testset_new['repo'] = []
#    testset_new['input_last_event_time'] = []
#    testset_new['gt_event_type'] = []
#    testset_new['gt_event_id'] = []
#    testset_new['gt_time_delay'] = []
#    testset_new['gt_user'] = []
#    testset_new['gt_user_cluster'] = []
#
#    X_test_new = []
#    for i in range(len(testset['X_test'])):  # for each repo-info
#        if len(testset['gt_event_type'][i]) == 0:
#            continue
#
#        X_test_new.append(testset['X_test'][i])
#
#        testset_new['repo'].append(
#                testset['repo'][i])
#
#        testset_new['input_last_event_time'].append(
#                testset['input_last_event_time'][i])
#
#        testset_new['gt_event_type'].append(
#                testset['gt_event_type'][i])
#
#        testset_new['gt_event_id'].append(
#                testset['gt_event_id'][i])
#
#        testset_new['gt_time_delay'].append(
#                testset['gt_time_delay'][i])
#
#        testset_new['gt_user'].append(
#                testset['gt_user'][i])
#
#        testset_new['gt_user_cluster'].append(
#                testset['gt_user_cluster'][i])
#
#    testset_new['X_test'] = np.array(X_test_new)
#    print_and_log(logger, "After post processing to remove ones "
#                  "are not in GT simulation cascades, "
#                  "X_test.shape: {}".format(
#                          testset_new['X_test'].shape))
#    # (675, 20, 269)
#    if (len(testset_new['X_test']) != len(testset_new['repo'])) or (
#            len(testset_new['X_test']) != len(
#                    testset_new['input_last_event_time'])) or (
#                    len(testset_new['X_test']) != len(
#                            testset_new['gt_event_type'])) or (
#                            len(testset_new['X_test']) != len
#                            (testset_new['gt_event_id'])) or (
#                                len(testset_new['X_test']) != len
#                                (testset_new['gt_time_delay'])):
#        print("new testset dimemsion mismatch!")
#        pdb.set_trace()
#    testset = testset_new

    # check num of 0.0 hours in the ground truth output
    gt_zero_hours_count = 0
    gt_hours_count = 0
    for i in range(len(testset['gt_time_delay'])):
        for j in range(len(testset['gt_time_delay'][i])):
            time_delay_hour = testset['gt_time_delay'][i][j]
            gt_hours_count += 1
            if time_delay_hour == 0:
                gt_zero_hours_count += 1
    print_and_log(
            logger,
            "Out of {} ground truth time delay values that the model "
            "needs to predict, {} of them are 0.0 time delay hour.".format(
                  gt_hours_count,
                  round(gt_zero_hours_count/gt_hours_count, 2)))

    # save testset
    print_and_log(logger, "save testset ...")

    testset_save_path = os.path.join(config['exp_save_dir'], "dataset",
                                     'testset.pickle')
    with open(testset_save_path, 'wb') as handle:
        pickle.dump(testset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print_and_log(logger, "testset.pickle save in {}".format(
            os.path.join(config['exp_save_dir'], "dataset")))

    with open(os.path.join(config['exp_save_dir'], "dataset",
                           'unique_repo_test.json'), 'w') as f:
        json.dump(unique_repo_test, f)
    with open(os.path.join(config['exp_save_dir'], "dataset",
                           'unique_user_test.json'), 'w') as f:
        json.dump(unique_user_test, f)

    print_and_log(logger, "the number of unique repos in "
                  "testing samples: {}".format(len(unique_repo_test)))
    print_and_log(logger, "the number of unique users in "
                  "testing samples: {}".format(len(unique_user_test)))

    print_and_log(logger, "{} took {}".format(
            "testset creation", time.time()-create_dataset_start))

    return testset


def testset_creation_given_gt(config, GT_avail=True):
    create_dataset_start = time.time()

    logger = set_logger(os.path.join(
            config['exp_save_dir'],
            'testset_creation_given_gt_' +
            dt.now().strftime("%Y-%m-%dT%H-%M-%SZ") + '.log'))
    print_and_log(logger, '{}'.format(
                'testset creation given gt...'))

    testset = dict()
    testset['X_test'] = list()
    testset['Y_test'] = list()
    testset['repo_list'] = list()
    testset['repo2sampleid'] = list()

    eventtype_2_id = config['eventtype_2_id']
    # id_2_eventtype = dict(zip(eventtype_2_id.values(),
    #                           eventtype_2_id.keys()))

    train_start = utc_timestamp(config['train_period']['start'])
    sim_start = utc_timestamp(config['sim_period']['start'])
    sim_end = utc_timestamp(config['sim_period']['end'])

    with open(os.path.join(config['exp_save_dir'], "dataset",
                           'unique_repo_train_vali.json'), 'r') as f:
        unique_repo_train_vali = json.load(f)
    repo_list = list(unique_repo_train_vali.keys())

    testset['repo_list'] = repo_list

    with open(config['github_event_repos_path'], 'rb') as f:
        github_event_repos = pickle.load(f)
    github_event_repos_set = set(
            [repo[:22]+'-'+repo[23:] for repo in github_event_repos])

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
                            '10log10_xplus1'))
            this_repo_profile_feat += repo_profile_raw[repo]['user_type']
            repo_profile_feat[repo] = this_repo_profile_feat
    if config['use_repo_idx_features']:
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
        padding_event_user_cluster = max(user_clusters.values()) + 1
        user_clusters[config['empty_user']] = padding_event_user_cluster
    if config['use_user_cluster_minmax']:
        with open(config['user_cluster_path'], 'r') as f:
            user_clusters = json.load(f)
        max_cluster_id = max(user_clusters.values())
        min_cluster_id = min(user_clusters.values())
        padding_event_user_cluster = max_cluster_id + 1
        max_cluster_id = padding_event_user_cluster
        max_minus_min = max_cluster_id - min_cluster_id
        if min_cluster_id != 0:
            print("min cluster id is not 0! Need to examine code!")
            pdb.set_trace()
        user_clusters[config['empty_user']] = padding_event_user_cluster
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
        cluster_profile_feat[padding_event_user_cluster] = [0] * config[
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

    user_has_no_cluster = set()

    sample_id = 0

    repo_y_et = []
    repo_y_td = []
    repo_y_uc = []

    print_and_log(logger, "gather testing data...")
    for repo in repo_list:  # for each cascade chain
        if (repo in config['repos_to_ignore']) or (
                repo in github_event_repos_set):
            # it is a event repo or repo should be ignore
            continue

        repo_X = []
        repo2sampleid = []

        one_chain_pd = load_jsongz_2_dataframe(
                os.path.join(config['cascade_dir'], repo + '.json.gz'))

        one_chain_pd = one_chain_pd.loc[
                (one_chain_pd['nodeTime'] >= train_start) &
                (one_chain_pd['nodeTime'] <= sim_end)]
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

#        (one_chain_event_new, one_chain_time_new, one_chain_user_new) = \
#            insert_no_event_for_a_chain_new(config,
#                                            one_chain_event, one_chain_time,
#                                            one_chain_user, sim_end+1)
        # if one_chain_event_new != one_chain_event:
        #     pdb.set_trace()

#        one_chain_event = one_chain_event_new
#        one_chain_time = one_chain_time_new
#        one_chain_user = one_chain_user_new

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

        # for each event sample in simulation period
        for i in range(config['window_size'], len(one_chain_event)):
            time_sample_outputevent = one_chain_time[i]
            event_sample_outputevent = one_chain_event[i]

            # if time_sample_outputevent in simulation period:
            # add this sample to testset
            if event_sample_outputevent == config['empty_event_type'] or (
                    event_sample_outputevent == eventtype_2_id['<soc>']):
                continue
            if one_chain_event[i-1] == eventtype_2_id['<soc>']:
                continue
            if not ((time_sample_outputevent >= sim_start) and (
                    time_sample_outputevent <= sim_end)):
                continue

            input_event_type = \
                one_chain_event[i-config['window_size']:i]
            # input_time = one_chain_time[i-config['window_size']:i]
            input_time_delay = \
                one_chain_time_delay[i-config['window_size']:i]
            input_user = one_chain_user[i-config['window_size']:i]
            input_cluster = []
            for user in input_user:
                try:
                    input_cluster.append(user_clusters[user])
                except KeyError:
                    user_has_no_cluster.add(user)
                    input_cluster.append(
                            padding_event_user_cluster)

            output_event_type = one_chain_event[i]
            output_time_delay = one_chain_time_delay[i]
            output_user = one_chain_user[i]
            try:
                output_cluster = user_clusters[output_user]
            except KeyError:
                user_has_no_cluster.add(output_user)
                output_cluster = padding_event_user_cluster

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
                    if input_cluster[j] == padding_event_user_cluster:
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
                    if input_user[j] == config['empty_user']:
                        x_j += [0] * config['dim_user_profile_features']
                    else:
                        try:
                            x_j += user_profile_feat[input_user[j]]
                        except KeyError:
                            x_j += [0] * config[
                                    'dim_user_profile_features']

                if config['use_user_activity_features']:
                    if input_user[j] == config['empty_user']:
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
                    if input_cluster[j] == padding_event_user_cluster:
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
                            x_j += [0] * config[
                                'dim_cluster_activity_features']

                if len(x_j) != x_dim:
                    print("len(x_j) != x_dim")
                    pdb.set_trace()
                x_vec.append(x_j)
            if len(x_vec) != config['window_size']:
                print("len(x_vec) != config['window_size']")
                pdb.set_trace()

            repo_X.append(x_vec)

            # output vec
            repo_y_et.append([output_event_type])
            repo_y_td.append([output_time_delay])
            repo_y_uc.append([output_cluster])

            repo2sampleid.append(sample_id)
            sample_id += 1
            # pdb.set_trace()

        # finish gathering test data for this repo
        testset['X_test'] += repo_X
        testset['repo2sampleid'].append(repo2sampleid)
        # pdb.set_trace()

    testset['X_test'] = np.array(testset['X_test'])
    testset['Y_test_et'] = np.array(repo_y_et)
    testset['Y_test_td'] = np.array(repo_y_td)
    testset['Y_test_uc'] = np.array(repo_y_uc)

    print_and_log(logger, "X_test.shape: {}".format(
                          testset['X_test'].shape))
    print_and_log(logger, "Y_test_et.shape: {}".format(
                          testset['Y_test_et'].shape))
    print_and_log(logger, "Y_test_td.shape: {}".format(
                          testset['Y_test_td'].shape))
    print_and_log(logger, "Y_test_uc.shape: {}".format(
                          testset['Y_test_uc'].shape))

    print_and_log(logger, "number of chains for testing: {}".format(
                          len(repo_list)))

    print_and_log(logger, "could not find cluster for {} users.".format(
            len(user_has_no_cluster)))

    if len(user_has_no_cluster) > 0:
        with open(os.path.join(
                config['exp_save_dir'], "dataset",
                'user_has_no_cluster_given_gt.pickle'), 'wb') as f:
            pickle.dump(user_has_no_cluster, f)

    print_and_log(logger, "the number of testing samples: {}".format(
            len(testset['X_test'])))

    with open(os.path.join(
            config['exp_save_dir'], "dataset",
            'testset_given_gt.pickle'), 'wb') as f:
        pickle.dump(testset, f)

    print_and_log(logger, "{} took {} min".format(
            "testset creation given gt",
            (time.time()-create_dataset_start)/60))
    # pdb.set_trace()
    return testset


def simulation_given_gt(config, model, testset):
    sim_took_time = time.time()

    # preparation
    result_save_path = os.path.join(config['exp_save_dir'],
                                    'test_given_gt_result-epoch{}'.format(
                                            config['load_model_epoch']))
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    eventtype_2_id = config['eventtype_2_id']
    id_2_eventtype = dict(zip(eventtype_2_id.values(),
                              eventtype_2_id.keys()))

    with open(config['user_cluster_path'], 'r') as f:
        user_clusters = json.load(f)
    padding_event_user_cluster = max(user_clusters.values()) + 1
    user_clusters[config['empty_user']] = padding_event_user_cluster

    # get predictions
    print("==========================================")
    print("start testing...")

    y_event_type = testset['Y_test_et']  # (3690, 1)
    y_time_delay = testset['Y_test_td']  # (3690, 1)
    y_user_cluster = testset['Y_test_uc']  # (3690, 1)

    [yhat_event_type, yhat_time_delay, yhat_user_cluster] = model.predict(
                testset['X_test'], verbose=0)
    # (3690, 12)  (3690, 1)  (3690, 101)

    pred_all_event_id = []
    pred_all_event_type = []
    pred_all_time_delay = []
    pred_all_user_cluster = []
    gt_all_event_id = []
    gt_all_event_type = []
    gt_all_time_delay = []
    gt_all_user_cluster = []

    chain_name = []

    for i in range(len(testset['repo_list'])):
        repo = testset['repo_list'][i]
        if len(testset['repo2sampleid'][i]) > 0:
            # start to simulate this chain
            pred_event_id = []
            pred_event_type = []
            pred_time_delay = []
            pred_user_cluster = []
            gt_event_id = []
            gt_event_type = []
            gt_time_delay = []
            gt_user_cluster = []

            chain_name.append(repo)

            print("testing {}/{}   repo: {}...".format(
                i+1, len(testset['repo_list']), repo))

            for sampleid in testset['repo2sampleid'][i]:
                # change one hot encoding into actual class
                predict_event = np.argmax(yhat_event_type[sampleid]) + 1

                # change time delay into actual time delay hour
                time_delay_hour = time_delay_normalization_reverse(
                            yhat_time_delay[sampleid][0],
                            config['time_delay_normalization_func'])

                # change one hot encoding into actual class
                predict_user_cluster = np.argmax(yhat_user_cluster[sampleid])

#                # modify for no_event_for_1month
#                if predict_event == eventtype_2_id['no_event_for_1month']:
#                    time_delay_hour = float(30*24)
#                    predict_user_cluster = user_clusters['no_event_user']
#
#                if time_delay_hour >= float(30*24):
#                    predict_event == eventtype_2_id['no_event_for_1month']
#                    time_delay_hour = float(30*24)
#                    predict_user_cluster = user_clusters['no_event_user']

                pred_event_id.append(
                        predict_event
                        )
                pred_event_type.append(
                        id_2_eventtype[predict_event]
                        )
                pred_time_delay.append(
                        round(time_delay_hour, 4)
                        )
                pred_user_cluster.append(
                        predict_user_cluster
                        )

                gt_event_id.append(
                        y_event_type[sampleid][0]
                        )
                gt_event_type.append(
                        id_2_eventtype[y_event_type[sampleid][0]]
                        )
                gt_time_delay.append(
                        round(time_delay_normalization_reverse(
                                y_time_delay[sampleid][0],
                                config['time_delay_normalization_func']), 4)
                        )
                gt_user_cluster.append(
                        y_user_cluster[sampleid][0]
                        )

            pred_all_event_id.append(pred_event_id)
            pred_all_event_type.append(pred_event_type)
            pred_all_time_delay.append(pred_time_delay)
            pred_all_user_cluster.append(pred_user_cluster)

            gt_all_event_id.append(gt_event_id)
            gt_all_event_type.append(gt_event_type)
            gt_all_time_delay.append(gt_time_delay)
            gt_all_user_cluster.append(gt_user_cluster)

            print("Testing Result:\n"
                  "Repo chain: {}\n"
                  "actual events: {}\n\n"
                  "predicted events: {}\n\n"
                  "actual time delay: {}\n\n"
                  "predicted time delay: {}\n\n"
                  "actual user cluster: {}\n\n"
                  "predicted user cluster: {}\n\n"
                  "==========================================\n".format(
                          repo,
                          gt_event_type,
                          pred_event_type,
                          gt_time_delay,
                          pred_time_delay,
                          gt_user_cluster,
                          pred_user_cluster
                          ))
            # pdb.set_trace()

    # pdb.set_trace()

    result = dict()
    result['chain_name'] = chain_name
    result['pred_all_event_id'] = pred_all_event_id
    result['pred_all_event_type'] = pred_all_event_type
    result['pred_all_time_delay'] = pred_all_time_delay
    result['pred_all_user_cluster'] = pred_all_user_cluster
    result['gt_all_event_id'] = gt_all_event_id
    result['gt_all_event_type'] = gt_all_event_type
    result['gt_all_time_delay'] = gt_all_time_delay
    result['gt_all_user_cluster'] = gt_all_user_cluster
    result['simulation_runningtime_inhours'] = (
            time.time() - sim_took_time)/3600

    with open(os.path.join(result_save_path,
                           'result.pickle'), 'wb') as handle:
        pickle.dump(result, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print('\n\result_give_gt.pickle saved in {}'.format(result_save_path))
    print("simulation given gt took {} hours".format(
            result['simulation_runningtime_inhours']))

    return
