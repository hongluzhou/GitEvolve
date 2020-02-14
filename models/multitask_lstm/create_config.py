#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:34:14 2019

@author: zhouhonglu
"""
import os


def create_config():
    config = {}
    config.update(create_date_config())
    config.update(create_path_config())
    config.update(create_dataset_config())
    config.update(create_training_config())
    config.update(create_testing_config())
    config.update(create_special_condition_config())
    config.update(create_evaluation_config())

    x_dim = 0
    if config['use_repo_embedding']:
        x_dim += config['dim_repo_embedding']

    if config['use_user_profile_features']:
        x_dim += config['dim_user_profile_features']

    if config['use_user_activity_features']:
        x_dim += config['dim_user_activity_features']

    if config['use_event_type_one_hot']:
        x_dim += len(config['eventtype_2_id'])

    if config['use_time_delay_features']:
        x_dim += config['dim_time_delay_features']

    config["x_dim"] = x_dim

    return config


def create_date_config():
    date_config = {
            # --- date
            'train_period': {'start': '2015-01-01T00:00:00Z',
                             'end': '2017-04-30T23:59:59Z'},
            'vali_period': {'start': '2017-05-01T00:00:00Z',
                            'end': '2017-05-31T23:59:59Z'},
            'gap_period': {'start': '2017-06-01T00:00:00Z',
                           'end': '2017-07-15T23:59:59Z'},
            'init_period': {'start': '2017-07-16T00:00:00Z',
                            'end': '2017-07-31T23:59:59Z'},
            'sim_period': {'start': '2017-08-01T00:00:00Z',
                           'end': '2017-08-31T23:59:59Z'}
            }
    return date_config


def create_path_config():
    # --- the root directory
    root_dir = '/media/data1/github_paper_exp'

    path_config = {
            # -- the directory to save experiment files
            'exp_save_dir': os.path.join(
                    root_dir, 'exp', 'myexp'),

            # --- the directory to cascades
            'cascade_dir': os.path.join(
                    root_dir, 'data',
                    'training_2015-01-01_2017-08-31'),

            # --- the directory to repo embedding files
            'embed_vec_path': os.path.join(
                    root_dir, 'data', 'repo_embeddings_userfocus.pickle'),

            # --- the directory to user profile feature files
            'user_profile_feat_path': os.path.join(
                    root_dir, 'data', 'features',
                    'users_profilefeat.json'),

            # --- the directory to user activity feature files
            'user_activity_feat_path': os.path.join(
                    root_dir, 'data', 'features',
                    'user_act_features.json'),

            # --- the directory to github event repo list
            'github_event_repos_path': os.path.join(
                    root_dir, 'data', 'event_repo_list.pickle'),

            'root_dir': root_dir
            }
    return path_config


def create_dataset_config():
    dataset_config = {
            'use_repo_embedding': True,
            'use_user_profile_features': True,
            'use_user_activity_features': True,
            'use_event_type_one_hot': True,
            'use_time_delay_features': True,

            'dim_repo_embedding': 256,
            'dim_user_profile_features': 255,
            'dim_user_activity_features': 28,
            'dim_time_delay_features': 1,

            # padding event type integer
            'empty_event_type': -1,
            'empty_time_delay': -1,
            'empty_user': 'padding_user',

            # time delay normalization function
            # choice: None, log10_xplus1, 10log10_xplus1
            'time_delay_normalization_func': '10log10_xplus1'}
    return dataset_config


def create_training_config():
    training_config = {
            # ---  map event type to a integer id
            'eventtype_2_id': {
                    'CreateEvent': 1,
                    'DeleteEvent': 2,
                    'ForkEvent': 3,
                    'PushEvent': 4,
                    'WatchEvent': 5,
                    'PullRequestEvent': 6,
                    'PullRequestReviewCommentEvent': 7,
                    'IssuesEvent': 8,
                    'IssueCommentEvent': 9,
                    'CommitCommentEvent': 10,
                    '<soc>': 11,
                    'no_event_for_1month': 12
                    },

            'window_size': 20,
            'batch_size': 256,
            'num_epochs': 800,
            'loss_weights': {'event_type_output': 0.8,
                             'time_delay_output': 1.0
                             },

            'num_gpu': 2,
            'multiprocessing_cpu': 40,

            'focalloss_gamma': 6,
            'focalloss_alpha': 0.6,

            'generator_shuffle': True,

            }
    return training_config


def create_testing_config():
    testing_config = {
            'load_model_epoch': 100,

            'keep_pred_max': 20,  # keep prediction threshold
            'keep_pred_round': 4
            }
    return testing_config


def create_special_condition_config():
    special_condition_config = {
            # ---  ignore certain repos such as super repo
            'repos_to_ignore': {
                    'ZxoNoLoGHJjegBQJ28V9fg-NsRTaZbKVhXc-ZEfBoeG3A'}}
    return special_condition_config


def create_evaluation_config():
    config = {
            'event_type_nlg_eval': True,
            'time_delay_overall_evaluation': True,

            'nlgeval_repo_dir': '/media/data1/github_paper_exp/nlg-eval/'
            }
    return config
