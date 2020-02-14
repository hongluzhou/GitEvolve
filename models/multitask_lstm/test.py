#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:39:11 2019

@author: zhouhonglu
"""
from models.multitask_lstm.utils import categorical_focal_loss
from models.multitask_lstm.utils import set_logger
from models.multitask_lstm.utils import print_and_log
from models.multitask_lstm.utils import time_delay_normalization
from models.multitask_lstm.utils import time_delay_normalization_reverse
from models.multitask_lstm.utils import load_jsongz_2_dataframe
from models.multitask_lstm.utils import insert_no_event_for_a_chain
from models.multitask_lstm.utils import insert_no_event_for_a_sim_GTchain
from models.multitask_lstm.utils import get_time_delay
from models.multitask_lstm.utils import utc_timestamp
# from models.multitask_lstm.utils import (
#         preprocess_cascade_files_to_remove_unnesseary_chains)

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
        testset = testset_creation(config)

    else:
        # load testset
        testset_save_path = os.path.join(config['exp_save_dir'], "dataset",
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

    model = keras.models.load_model(trained_model_path,
                                    custom_objects=custom_object)

    # simulation
    simulation(config, model, testset)

    return


def simulation(config, model, testset):
    sim_took_time = time.time()

    print("simultion will take ground truth user and chain length as input.")
    print("if we predict a chain with length longer than its ground truth, "
          "we will cut it off to make sure all the predicted chains "
          "whose length are less or equal to ground truth.")

    # preparation
    result_save_path = os.path.join(config['exp_save_dir'],
                                    'test_result-epoch{}'.format(
                                            config['load_model_epoch']))
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    keep_pred_max = config['keep_pred_max']
    print("keep prediction threshold: {}".format(keep_pred_max))

    eventtype_2_id = config['eventtype_2_id']
    reversed_dictionary = dict(zip(eventtype_2_id.values(),
                                   eventtype_2_id.keys()))

    test_end_utc = utc_timestamp(config['sim_period']['end'])

    # get x_dim
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
    chains_applied_keep_pred = []

    # for each repo-info in the testset
    for i in range(len(testset['X_test'])):
        repo_info = testset['repo_info'][i]
        print("testing {}/{}   repo: {}  info: {}...".format(
                i+1, len(testset['X_test']),
                repo_info[:45],
                repo_info[45:]))
        repo = repo_info[:45]

        # get ground truth chain length
        gt_chain_length = len(testset['gt_user'][i])

        keep_pred = dict()
        keep_pred_cond_pass = True
        applied = False
        cut_off = False

        # get the largest time delay this chain could have
        largest_time_delay_hours = \
            (test_end_utc - testset['input_last_event_time'][i])/3600
        test_time_delay_total = 0

        # start to simulate this chain
        pred_event_id = []
        pred_event_type = []
        pred_time_delay = []

        # predict the first event
        sample_input_this_chain = np.zeros((
                1, config['window_size'], x_dim))
        sample_next_this_chain = np.zeros((
                1, config['window_size'], x_dim))

        sample_input_this_chain[0, :] = testset['X_test'][i]
        sample_next_this_chain[0, :config['window_size']-1, :] = \
            sample_input_this_chain[0, 1:, :]

        embed_vec = embed_vec_all[repo_info[:22]+'/'+repo_info[23:45]]

        (y_pred, time_delay) = model.predict(
                sample_input_this_chain, batch_size=1, verbose=0)

        # change one hot encoding output event into actual event id
        predict_event = np.argmax(y_pred[0]) + 1

        # modify time_delay and time_delay_hour
        if predict_event == eventtype_2_id['no_event_for_1month']:
            time_delay_hour = float(30*24)
            if config['time_delay_normalization_func'] is not None:
                time_delay = time_delay_normalization(
                        time_delay_hour,
                        config['time_delay_normalization_func'])
            else:
                print("not implemented!")
                pdb.set_trace()

        else:
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

        # pdb.set_trace()
        if (test_time_delay_total + time_delay_hour <=
                largest_time_delay_hours) and (keep_pred_cond_pass):
            # add predicted new event
            pred_event_id.append(predict_event)
            pred_event_type.append(reversed_dictionary[predict_event])
            pred_time_delay.append(time_delay_hour)
            test_time_delay_total += time_delay_hour

            gt_user = testset['gt_user'][i][len(pred_event_id)-1]
            # feature calculation for predicted new event
            dim_pointer = 0

            if config['use_repo_embedding']:
                sample_next_this_chain[0, -1, :dim_pointer + config[
                        'dim_repo_embedding']] = embed_vec
                dim_pointer += config['dim_repo_embedding']

            if config['use_user_profile_features']:
                if gt_user == config['empty_user'] or (
                        gt_user == 'no_event_user'):
                    sample_next_this_chain[
                            0, -1, dim_pointer:dim_pointer + config[
                                    'dim_user_profile_features']] = \
                                [0] * config['dim_user_profile_features']
                else:
                    try:
                        sample_next_this_chain[
                            0, -1, dim_pointer:dim_pointer + config[
                                    'dim_user_profile_features']] = \
                                    user_profile_feat[gt_user]
                    except KeyError:
                        sample_next_this_chain[
                            0, -1, dim_pointer:dim_pointer + config[
                                    'dim_user_profile_features']] = \
                                    [0] * config['dim_user_profile_features']
                dim_pointer += config['dim_user_profile_features']

            if config['use_user_activity_features']:
                if gt_user == config['empty_user'] or (
                        gt_user == 'no_event_user'):
                    sample_next_this_chain[
                            0, -1, dim_pointer:dim_pointer + config[
                                    'dim_user_activity_features']] = \
                                [0] * config['dim_user_activity_features']
                else:
                    try:
                        thisrepo_feat = \
                            user_act_feat[gt_user][repo]
                    except KeyError:
                        # this user-repo no event in training period
                        thisrepo_feat = \
                            [0] * int(config[
                                    'dim_user_activity_features']/2)
                    allrepo_feat = \
                        user_act_feat[gt_user]['all']
                    sample_next_this_chain[
                            0, -1, dim_pointer:dim_pointer + config[
                                    'dim_user_activity_features']] = (
                            thisrepo_feat + allrepo_feat)
                dim_pointer += config['dim_user_activity_features']

            if config['use_event_type_one_hot']:
                one_hot = [0] * len(config['eventtype_2_id'])
                one_hot[predict_event-1] = 1
                sample_next_this_chain[0, -1, dim_pointer:dim_pointer + len(
                                    config['eventtype_2_id'])] = \
                    np.array(one_hot)
                dim_pointer += len(config['eventtype_2_id'])

            if config['use_time_delay_features']:
                sample_next_this_chain[
                        0, -1, dim_pointer: dim_pointer + config[
                                'dim_time_delay_features']] = time_delay
                dim_pointer += config['dim_time_delay_features']

            if dim_pointer != x_dim:
                print("dim_pointer != x_dim")
                pdb.set_trace()

            # print("start testing, {}/{}, {}, {}, {}, {}".format(
            #         i, len(testset['X_test']),
            #         reversed_dictionary[predict_event],
            #         time_delay_hour, test_time_delay_total,
            #         largest_time_delay_hours))
            # pdb.set_trace()
            while ((test_time_delay_total +
                    time_delay_hour) <= largest_time_delay_hours) and (
                    keep_pred_cond_pass) and (
                            not cut_off):

                (y_pred, time_delay) = model.predict(
                    sample_next_this_chain, batch_size=1, verbose=0)

                # change one hot encoding output event into actual event id
                predict_event = np.argmax(y_pred[0]) + 1

                # modify time_delay and time_delay_hour or keep_pred_cond_pass
                if predict_event == eventtype_2_id['no_event_for_1month']:
                    time_delay_hour = float(30*24)
                    if config['time_delay_normalization_func'] is not None:
                        time_delay = time_delay_normalization(
                                time_delay_hour,
                                config['time_delay_normalization_func'])
                        keep_pred_cond_pass = True
                    else:
                        print("not implemented!")
                        pdb.set_trace()
                else:
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

                # update sample_next_this_chain
                sample_next_this_chain_new = np.zeros((
                        1, config['window_size'], x_dim))
                sample_next_this_chain_new[
                        0, :config['window_size']-1, :] = \
                    sample_next_this_chain[0, 1:, :]
                sample_next_this_chain = sample_next_this_chain_new

                if len(pred_event_id) >= gt_chain_length:
                        cut_off = True

                if ((test_time_delay_total +
                     time_delay_hour) <= largest_time_delay_hours) and (
                        keep_pred_cond_pass) and (not cut_off):
                    # add predicted new event
                    pred_event_id.append(predict_event)
                    pred_event_type.append(
                            reversed_dictionary[predict_event])
                    pred_time_delay.append(time_delay_hour)
                    test_time_delay_total += time_delay_hour

                    gt_user = testset['gt_user'][i][len(pred_event_id)-1]

                    # feature calculation for predicted new event node
                    dim_pointer = 0

                    if config['use_repo_embedding']:
                        sample_next_this_chain[0, -1, :dim_pointer + config[
                                'dim_repo_embedding']] = embed_vec
                        dim_pointer += config['dim_repo_embedding']

                    if config['use_user_profile_features']:
                        if gt_user == config['empty_user'] or (
                                gt_user == 'no_event_user'):
                            sample_next_this_chain[
                                    0, -1, dim_pointer:dim_pointer + config[
                                            'dim_user_profile_features']] = \
                                        [0] * config[
                                                'dim_user_profile_features']
                        else:
                            try:
                                sample_next_this_chain[
                                    0, -1, dim_pointer:dim_pointer + config[
                                            'dim_user_profile_features']] = \
                                            user_profile_feat[gt_user]
                            except KeyError:
                                sample_next_this_chain[
                                    0, -1, dim_pointer:dim_pointer + config[
                                            'dim_user_profile_features']
                                    ] = [0] * config[
                                            'dim_user_profile_features']
                        dim_pointer += config['dim_user_profile_features']

                    if config['use_user_activity_features']:
                        if gt_user == config['empty_user'] or (
                                gt_user == 'no_event_user'):
                            sample_next_this_chain[
                                    0, -1, dim_pointer:dim_pointer + config[
                                            'dim_user_activity_features']
                                    ] = [0] * config[
                                            'dim_user_activity_features']
                        else:
                            try:
                                thisrepo_feat = \
                                    user_act_feat[gt_user][repo]
                            except KeyError:
                                # this user-repo no event in training period
                                thisrepo_feat = \
                                    [0] * int(config[
                                            'dim_user_activity_features']/2)
                            allrepo_feat = \
                                user_act_feat[gt_user]['all']
                            sample_next_this_chain[
                                    0, -1, dim_pointer:dim_pointer + config[
                                            'dim_user_activity_features']
                                    ] = (thisrepo_feat + allrepo_feat)
                        dim_pointer += config['dim_user_activity_features']

                    if config['use_event_type_one_hot']:
                        one_hot = [0] * len(config['eventtype_2_id'])
                        one_hot[predict_event-1] = 1
                        sample_next_this_chain[
                                0, -1, dim_pointer:dim_pointer + len(
                                        config['eventtype_2_id'])] = \
                            np.array(one_hot)
                        dim_pointer += len(config['eventtype_2_id'])

                    if config['use_time_delay_features']:
                        sample_next_this_chain[
                                0, -1, dim_pointer: dim_pointer + config[
                                        'dim_time_delay_features']] = \
                                time_delay
                        dim_pointer += config['dim_time_delay_features']

                    if dim_pointer != x_dim:
                        print("dim_pointer != x_dim")
                        pdb.set_trace()

                    # print(repo_info)
                    # print(sample_next_this_chain)
                    # print(time_delay, time_delay_hour, predict_event)
                    # pdb.set_trace()

                    # print("start testing, {}/{}, {}, {}, {}, {}, "
                    #       "{}, {}".format(
                    #         i, len(testset['X_test']),
                    #         reversed_dictionary[predict_event],
                    #         round(time_delay_hour, 2),
                    #         round(test_time_delay_total, 2),
                    #         round(largest_time_delay_hours, 2),
                    #         len(pred_event_id),
                    #         len(testset['gt_time_delay'][i])))

        # pdb.set_trace()
        pred_all_event_id.append(pred_event_id)
        pred_all_event_type.append(pred_event_type)
        pred_all_time_delay.append(pred_time_delay)
        if applied:
            chains_applied_keep_pred.append(repo_info)
        print("Testing Result:\n"
              "Info: {}\n"
              "Chain: {}\n"
              "ground truth chain length: {}\n"
              "predicted chain length: {}\n"
              "time delay DTW: {}\n"
              "applied keep prediction threshold? {} (threshold {})\n"
              "cut off: {}\n\n"
              "actual events: {}\n\n"
              "predicted events: {}\n\n"
              "actual time delay: {}\n\n"
              "predicted time delay: {}\n\n"
              "==========================================\n".format(
                      repo_info[45:],
                      repo_info[:45],
                      len(testset['gt_time_delay'][i]),
                      len(pred_event_id),
                      fastdtw(testset['gt_time_delay'][i],
                              pred_time_delay)[0],
                      applied,
                      keep_pred_max,
                      cut_off,
                      testset['gt_event_type'][i],
                      pred_event_type,
                      testset['gt_time_delay'][i],
                      pred_time_delay
                      ))
        # pdb.set_trace()

    result = dict()
    result['chain_name'] = testset['repo_info']
    result['pred_all_event_id'] = pred_all_event_id
    result['pred_all_event_type'] = pred_all_event_type
    result['pred_all_time_delay'] = pred_all_time_delay
    result['gt_all_event_id'] = testset['gt_event_id']
    result['gt_all_event_type'] = testset['gt_event_type']
    result['gt_all_time_delay'] = testset['gt_time_delay']
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


def simulation_no_user_feat(config, model, testset):
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
    reversed_dictionary = dict(zip(eventtype_2_id.values(),
                                   eventtype_2_id.keys()))

    test_end_utc = utc_timestamp(config['sim_period']['end'])

    # get x_dim
    x_dim = config['x_dim']
    if config['use_repo_embedding']:
        with open(config['embed_vec_path'], 'rb') as f:
            embed_vec_all = pickle.load(f)
    if config['use_user_profile_features']:
        pass
    if config['use_user_activity_features']:
        pass
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
    chains_applied_keep_pred = []

    # for each repo-info in the testset
    for i in range(len(testset['X_test'])):
        repo_info = testset['repo_info'][i]
        print("testing {}/{}   repo: {}  info: {}...".format(
                i+1, len(testset['X_test']),
                repo_info[:45],
                repo_info[45:]))

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

        # predict the first event
        sample_input_this_chain = np.zeros((
                1, config['window_size'], x_dim))
        sample_next_this_chain = np.zeros((
                1, config['window_size'], x_dim))

        sample_input_this_chain[0, :] = testset['X_test'][i]
        sample_next_this_chain[0, :config['window_size']-1, :] = \
            sample_input_this_chain[0, 1:, :]

        embed_vec = embed_vec_all[repo_info[:22]+'/'+repo_info[23:45]]

        (y_pred, time_delay) = model.predict(
                sample_input_this_chain, batch_size=1, verbose=0)

        # change one hot encoding output event into actual event id
        predict_event = np.argmax(y_pred[0]) + 1

        # modify time_delay and time_delay_hour
        if predict_event == eventtype_2_id['no_event_for_1month']:
            time_delay_hour = float(30*24)
            if config['time_delay_normalization_func'] is not None:
                time_delay = time_delay_normalization(
                        time_delay_hour,
                        config['time_delay_normalization_func'])
            else:
                print("not implemented!")
                pdb.set_trace()

        else:
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

        # pdb.set_trace()
        if (test_time_delay_total + time_delay_hour <=
                largest_time_delay_hours) and (keep_pred_cond_pass):
            # add predicted new event
            pred_event_id.append(predict_event)
            pred_event_type.append(reversed_dictionary[predict_event])
            pred_time_delay.append(time_delay_hour)
            test_time_delay_total += time_delay_hour

            # feature calculation for predicted new event
            if config['use_repo_embedding']:
                sample_next_this_chain[0, -1, :config[
                        'dim_repo_embedding']] = embed_vec
            if config['use_user_profile_features']:
                pass
            if config['use_user_activity_features']:
                pass
            if config['use_event_type_one_hot']:
                one_hot = [0] * len(config['eventtype_2_id'])
                one_hot[predict_event-1] = 1
                sample_next_this_chain[0, -1, config[
                        'dim_repo_embedding']: config[
                                'dim_repo_embedding'] + len(
                                    config['eventtype_2_id'])] = \
                    np.array(one_hot)
            if config['use_time_delay_features']:
                sample_next_this_chain[0, -1, config[
                        'dim_repo_embedding'] + len(config[
                                'eventtype_2_id']): config[
                                'dim_repo_embedding'] + len(
                                    config['eventtype_2_id']) + config[
                                    'dim_time_delay_features']] = time_delay
            # print("start testing, {}/{}, {}, {}, {}, {}".format(
            #         i, len(testset['X_test']),
            #         reversed_dictionary[predict_event],
            #         time_delay_hour, test_time_delay_total,
            #         largest_time_delay_hours))
            # pdb.set_trace()
            while ((test_time_delay_total +
                    time_delay_hour) <= largest_time_delay_hours) and (
                    keep_pred_cond_pass):
                (y_pred, time_delay) = model.predict(
                    sample_next_this_chain, batch_size=1, verbose=0)

                # change one hot encoding output event into actual event id
                predict_event = np.argmax(y_pred[0]) + 1

                # modify time_delay and time_delay_hour or keep_pred_cond_pass
                if predict_event == eventtype_2_id['no_event_for_1month']:
                    time_delay_hour = float(30*24)
                    if config['time_delay_normalization_func'] is not None:
                        time_delay = time_delay_normalization(
                                time_delay_hour,
                                config['time_delay_normalization_func'])
                        keep_pred_cond_pass = True
                    else:
                        print("not implemented!")
                        pdb.set_trace()
                else:
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
                            reversed_dictionary[predict_event])
                    pred_time_delay.append(time_delay_hour)
                    test_time_delay_total += time_delay_hour

                    # feature calculation for predicted new event node
                    if config['use_repo_embedding']:
                        sample_next_this_chain[0, -1, :config[
                                'dim_repo_embedding']] = embed_vec
                    if config['use_user_profile_features']:
                        pass
                    if config['use_user_activity_features']:
                        pass
                    if config['use_event_type_one_hot']:
                        one_hot = [0] * len(config['eventtype_2_id'])
                        one_hot[predict_event-1] = 1
                        sample_next_this_chain[0, -1, config[
                                'dim_repo_embedding']: config[
                                        'dim_repo_embedding'] + len(
                                            config['eventtype_2_id'])] = \
                            np.array(one_hot)
                    if config['use_time_delay_features']:
                        sample_next_this_chain[0, -1, config[
                                'dim_repo_embedding'] + len(config[
                                        'eventtype_2_id']): config[
                                        'dim_repo_embedding'] + len(
                                            config['eventtype_2_id']) + config[
                                            'dim_time_delay_features']
                                            ] = time_delay

                    # print(repo_info)
                    # print(sample_next_this_chain)
                    # print(time_delay, time_delay_hour, predict_event)
                    # pdb.set_trace()

                    # print("start testing, {}/{}, {}, {}, {}, {}, "
                    #       "{}, {}".format(
                    #         i, len(testset['X_test']),
                    #         reversed_dictionary[predict_event],
                    #         round(time_delay_hour, 2),
                    #         round(test_time_delay_total, 2),
                    #         round(largest_time_delay_hours, 2),
                    #         len(pred_event_id),
                    #         len(testset['gt_time_delay'][i])))

        # pdb.set_trace()
        pred_all_event_id.append(pred_event_id)
        pred_all_event_type.append(pred_event_type)
        pred_all_time_delay.append(pred_time_delay)
        if applied:
            chains_applied_keep_pred.append(repo_info)
        print("Testing Result:\n"
              "Info: {}\n"
              "Chain: {}\n"
              "ground truth chain length: {}\n"
              "predicted chain length: {}\n"
              "time delay DTW: {}\n"
              "applied keep prediction threshold? {} (threshold {})\n\n"
              "actual events: {}\n\n"
              "predicted events: {}\n\n"
              "actual time delay: {}\n\n"
              "predicted time delay: {}\n\n"
              "==========================================\n".format(
                      repo_info[45:],
                      repo_info[:45],
                      len(testset['gt_time_delay'][i]),
                      len(pred_event_id),
                      fastdtw(testset['gt_time_delay'][i],
                              pred_time_delay)[0],
                      applied,
                      keep_pred_max,
                      testset['gt_event_type'][i],
                      pred_event_type,
                      testset['gt_time_delay'][i],
                      pred_time_delay
                      ))
        # pdb.set_trace()

    result = dict()
    result['chain_name'] = testset['repo_info']
    result['pred_all_event_id'] = pred_all_event_id
    result['pred_all_event_type'] = pred_all_event_type
    result['pred_all_time_delay'] = pred_all_time_delay
    result['gt_all_event_id'] = testset['gt_event_id']
    result['gt_all_event_type'] = testset['gt_event_type']
    result['gt_all_time_delay'] = testset['gt_time_delay']
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
                                     'testset_creation' +
                                     dt.now().strftime("%Y-%m-%dT%H-%M-%SZ") +
                                     '.log'))
    print_and_log(logger, '{}'.format(
                'testset creation...'))

    testset = dict()
    testset['X_test'] = []
    testset['repo_info'] = []
    testset['input_last_event_time'] = []
    testset['gt_event_type'] = None
    testset['gt_event_id'] = None
    testset['gt_time_delay'] = None
    testset['gt_user'] = None
    testset['infoid2foldername'] = None
    testset['foldername2infoid'] = None
    X_test = []

    eventtype_2_id = config['eventtype_2_id']
    reversed_dictionary = dict(zip(eventtype_2_id.values(),
                                   eventtype_2_id.keys()))

    # init_start = utc_timestamp(config['init_period']['start'])
    init_end = utc_timestamp(config['init_period']['end'])
    sim_start = utc_timestamp(config['sim_period']['start'])
    # sim_end = utc_timestamp(config['sim_period']['end'])

    with open(config['github_event_repos_path'], 'rb') as f:
        github_event_repos = pickle.load(f)
    github_event_repos_set = set(
            [repo[:22]+'-'+repo[23:] for repo in github_event_repos])

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

    cascades_path = os.path.join(config['cascade_dir'], "initialization",
                                 "github")
    sim_info_ids = [f for f in os.listdir(cascades_path) if "id_" in f]

    # preprocess_cascade_files_to_remove_unnesseary_chains(
    #     config, cascades_path, github_event_repos_set)
    # pdb.set_trace()

    with open(os.path.join(
            cascades_path, "infoid2foldername.json"), 'r') as f:
        infoid2foldername = json.load(f)
    testset['infoid2foldername'] = infoid2foldername

    foldername2infoid = dict()
    for infoid in infoid2foldername:
        foldername2infoid[infoid2foldername[infoid]] = infoid
    with open(os.path.join(
            cascades_path, "foldername2infoid.json"), 'w') as f:
        json.dump(foldername2infoid, f)
    testset['foldername2infoid'] = foldername2infoid

    for i in range(len(sim_info_ids)):
        folder_name = sim_info_ids[i]
        sim_info_ids[i] = foldername2infoid[folder_name]

    unique_repo_test = dict()
    unique_user_test = dict()

    # gather simulation sample input
    sim_info_ids = sorted(sim_info_ids)
    print_and_log(logger, "gather simulation sample input...")
    for info_id in sim_info_ids:
        filelist = [f for f in os.listdir(os.path.join(
                        cascades_path, infoid2foldername[info_id])
                        ) if '.json.gz' in f]
        filelist = sorted(filelist)
        for file in filelist:  # for each cascade chain
            if (file[:-8] in config['repos_to_ignore']) or (
                    file[:-8] in github_event_repos_set):
                # it is a event repo or repo should be ignore
                print("it is a event repo or repo should be ignore, "
                      "shouldn't happen!")
                pdb.set_trace()

            one_chain_pd = load_jsongz_2_dataframe(
                    os.path.join(cascades_path,
                                 infoid2foldername[info_id], file))
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

            """
            init chains have all events before sim
            """
            if max(one_chain_time) > init_end:
                print("max(one_chain_time) > init_end")
                pdb.set_trace()

            (one_chain_event_new, one_chain_time_new,
             one_chain_user_new) = \
                insert_no_event_for_a_chain(config,
                                            one_chain_event,
                                            one_chain_time,
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

            testset['input_last_event_time'].append(
                    one_chain_time[-1])
            testset['repo_info'].append(file[:-8]+info_id)

            input_event_type = one_chain_event[-config['window_size']:]
            input_time_delay = one_chain_time_delay[-config['window_size']:]
            input_user = one_chain_user[i-config['window_size']:i]

            # initialize input vector, and output vector for this sample
            x_vec = []

            # load node embeding vector
            if config['use_repo_embedding']:
                chain_temp = file[:-8][:22] + '/' + file[:-8][23:]
                try:
                    embed_vec = np.array(embed_vec_all[chain_temp])
                except KeyError:
                    print_and_log(logger, "Could not find embedding vector "
                                  "for {}!".format(chain_temp))
                    pdb.set_trace()
                    # embed_vec = np.random.rand(1, config["embed_vec_len"])

            # input feature vector
            for j in range(len(input_event_type)):  # for each event node
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
                                user_act_feat[input_user[j]][file[:-8]]
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

                x_vec.append(x_j)

            X_test.append(x_vec)

    testset['X_test'] = np.array(X_test)

    print_and_log(logger, "X_test.shape: {}".format(testset['X_test'].shape))
    # (1093, 20, 269)

    # gather simulation sample output
    if GT_avail:
        print_and_log(logger, "ground truth available. \n"
                      "gather simulation sample output...")
        testset['gt_event_type'] = []
        testset['gt_time_delay'] = []
        testset['gt_event_id'] = []
        testset['gt_user'] = []

        GT_cascades_path = os.path.join(config['cascade_dir'], "simulation",
                                        "github")
        GT_sim_info_ids = [f for f in os.listdir(GT_cascades_path)
                           if "id_" in f]

        with open(os.path.join(
                GT_cascades_path, "infoid2foldername.json"), 'r') as f:
            GT_infoid2foldername = json.load(f)

        GT_foldername2infoid = dict()
        for infoid in GT_infoid2foldername:
            GT_foldername2infoid[GT_infoid2foldername[infoid]] = infoid
        with open(os.path.join(
                GT_cascades_path, "foldername2infoid.json"), 'w') as f:
            json.dump(GT_foldername2infoid, f)

        for i in range(len(GT_sim_info_ids)):
            GT_folder_name = GT_sim_info_ids[i]
            GT_sim_info_ids[i] = GT_foldername2infoid[GT_folder_name]

        if len(sim_info_ids) != len(GT_sim_info_ids):
            print_and_log(logger,
                          "len(sim_info_ids) != len(GT_sim_info_ids)!")
            print_and_log(logger,
                          "len(sim_info_ids): {}".format(
                                  len(set(sim_info_ids))))  # 1040
            print_and_log(logger,
                          "len(GT_sim_info_ids): {}".format(
                                  len(set(GT_sim_info_ids))))  # 628
            print_and_log(logger,
                          "{} info_ids in init but not in sim.".format(
                                  len([infotem for infotem in sim_info_ids
                                       if infotem not in GT_sim_info_ids]))
                          )  # 441
            print_and_log(logger,
                          "{} info_ids in sim but not in init.".format(
                                  len([infotem for infotem in GT_sim_info_ids
                                       if infotem not in sim_info_ids]))
                          )  # 29

        # for each gathered simulation sample input
        for i in range(len(testset['repo_info'])):
            repo_info = testset['repo_info'][i]
            repo_this_sample = repo_info[:45]
            info_this_sample = repo_info[45:]

            if info_this_sample not in GT_sim_info_ids:
                # this info_id (sample chain) has no events in testing period
                testset['gt_event_id'].append([])
                testset['gt_event_type'].append([])
                testset['gt_time_delay'].append([])
                testset['gt_user'].append([])
            else:
                # get all chain_ids this info_id has in testing period
                GT_chainlist_this_info = [file[:-8] for file in os.listdir(
                        os.path.join(GT_cascades_path,
                                     GT_infoid2foldername[info_this_sample])
                        ) if '.json.gz' in file]
                if repo_this_sample not in set(GT_chainlist_this_info):
                    # this sample chain has no events in testing period
                    testset['gt_event_id'].append([])
                    testset['gt_event_type'].append([])
                    testset['gt_time_delay'].append([])
                    testset['gt_user'].append([])
                else:
                    # this sample chain has events in testing period
                    one_chain_pd = load_jsongz_2_dataframe(
                            os.path.join(GT_cascades_path,
                                         GT_infoid2foldername[
                                                 info_this_sample],
                                         repo_this_sample+'.json.gz'))
                    one_chain_pd = one_chain_pd.sort_values(by=['nodeTime'])

                    unique_repo_test[repo_this_sample] = []
                    for user in one_chain_pd['nodeUserID']:
                        unique_user_test[user] = []

                    one_chain_event_type = []
                    one_chain_event_id = []
                    one_chain_time = []
                    one_chain_user = []
                    one_chain_event_type += [event for event in one_chain_pd[
                            'actionType']]
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

                    (one_chain_event_new, one_chain_time_new,
                     one_chain_user_new) = \
                        insert_no_event_for_a_sim_GTchain(
                                config,
                                one_chain_event_id, one_chain_time,
                                input_last_event_time_this_chain,
                                one_chain_user)

                    # if one_chain_event_new != one_chain_event:
                    #     pdb.set_trace()

                    one_chain_event_id = one_chain_event_new
                    one_chain_time = one_chain_time_new
                    one_chain_user = one_chain_user_new
                    one_chain_event_type = [reversed_dictionary[
                            event] for event in one_chain_event_id]

                    # calculate time delay sequence
                    one_chain_time_delay = []
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

                    # pdb.set_trace()

    # post-process testset to make prediction only for chains gt have events
    if GT_avail:
        testset_new = dict()
        testset_new['repo_info'] = []
        testset_new['input_last_event_time'] = []
        testset_new['gt_event_type'] = []
        testset_new['gt_event_id'] = []
        testset_new['gt_time_delay'] = []
        testset_new['gt_user'] = []
        testset_new['infoid2foldername'] = testset['infoid2foldername']
        testset_new['foldername2infoid'] = testset['foldername2infoid']

        X_test_new = []
        for i in range(len(testset['X_test'])):  # for each repo-info
            if len(testset['gt_event_type'][i]) == 0:
                continue

            X_test_new.append(testset['X_test'][i])

            testset_new['repo_info'].append(
                    testset['repo_info'][i])

            testset_new['input_last_event_time'].append(
                    testset['input_last_event_time'][i])

            testset_new['gt_event_type'].append(
                    testset['gt_event_type'][i])

            testset_new['gt_event_id'].append(
                    testset['gt_event_id'][i])

            testset_new['gt_time_delay'].append(
                    testset['gt_time_delay'][i])

            testset_new['gt_user'].append(
                    testset['gt_user'][i])

        testset_new['X_test'] = np.array(X_test_new)
        print_and_log(logger, "After post processing to remove ones "
                      "are not in GT simulation cascades, "
                      "X_test.shape: {}".format(
                              testset_new['X_test'].shape))
        # (675, 20, 269)
        if (len(testset_new['X_test']) != len(testset_new['repo_info'])) or (
                len(testset_new['X_test']) != len(
                        testset_new['input_last_event_time'])) or (
                        len(testset_new['X_test']) != len(
                                testset_new['gt_event_type'])) or (
                                len(testset_new['X_test']) != len
                                (testset_new['gt_event_id'])) or (
                                    len(testset_new['X_test']) != len
                                    (testset_new['gt_time_delay'])):
            print("new testset dimemsion mismatch!")
            pdb.set_trace()
        testset = testset_new

    # check num of 0.0 hours in the ground truth output
    gt_zero_hours_count = 0
    gt_hours_count = 0
    for i in range(len(testset['gt_time_delay'])):
        for j in range(len(testset['gt_time_delay'][i])):
            time_delay_hour = testset['gt_time_delay'][i][j]
            gt_hours_count += 1
            if time_delay_hour == 0:
                gt_zero_hours_count += 1
    print("Out of {} ground truth time delay values that the model "
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
