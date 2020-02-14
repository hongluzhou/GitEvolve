#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:34:14 2019

@author: zhouhonglu
"""
import os
import multiprocessing


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

    if config['use_repo_profile_features']:
        x_dim += config['dim_repo_profile_features']

    if config['use_repo_idx_features']:
        x_dim += config['dim_repo_idx_features']

    if config['use_repo_activity_features']:
        x_dim += config['dim_repo_activity_features']

    if config['use_user_profile_features']:
        x_dim += config['dim_user_profile_features']

    if config['use_user_activity_features']:
        x_dim += config['dim_user_activity_features']

    if config['use_event_type_one_hot']:
        x_dim += len(config['eventtype_2_id'])

    if config['use_time_delay_features']:
        x_dim += config['dim_time_delay_features']

    if config['use_user_cluster_one_hot']:
        x_dim += config['dim_user_cluster_one_hot']

    if config['use_user_cluster_minmax']:
        x_dim += config['dim_user_cluster_minmax']

    if config['use_cluster_profile_features']:
        x_dim += config['dim_cluster_profile_features']

    if config['use_cluster_activity_features']:
        x_dim += config['dim_cluster_activity_features']

    config["x_dim"] = x_dim

    if config['multiprocessing_cpu'] == 'all':
        config['multiprocessing_cpu'] = multiprocessing.cpu_count()

    return config


def create_date_config():
    date_config = {
            # --- date
            'train_period': {'start': '2015-01-01T00:00:00Z',
                             'end': '2017-07-31T23:59:59Z'},
            'vali_period': {'start': '2017-08-01T00:00:00Z',
                            'end': '2017-08-15T23:59:59Z'},
            'sim_period': {'start': '2017-08-16T00:00:00Z',
                           'end': '2017-08-31T23:59:59Z'}
            }
    return date_config


def create_path_config():
    # --- the root directory
    root_dir = '/media/data1/github_paper_exp'

    path_config = {
            # -- the directory to save experiment files
            'exp_save_dir': os.path.join(
                    root_dir, 'exp', 'www_review_repeat_last_action'),  # aug7_2_vanilla

            # --- the directory to cascades
            'cascade_dir': os.path.join(
                    root_dir, 'data',
                    'training_2015-01-01_2017-08-31'),

            # --- the directory to repo embedding files
            'embed_vec_path': os.path.join(
                    root_dir, 'data', 'repo_embeddings_userfocus.pickle'),

            # --- the directory to repo profile feat
            'repo_profile_feat_path': os.path.join(
                    root_dir, 'data', 'repo_profilefeat_lowdim.json'),

            # --- the directory to repo activity feature files
            'repo_activity_feat_path': os.path.join(
                    root_dir, 'data',
                    'repo_activity_feat_aug9_10log10_xplus1_normalization_' +
                    '2015-01-01_2017-07-31',
                    'repo_act_features.pickle'),

            # --- the directory to user profile feature files
            'user_profile_feat_path': os.path.join(
                    root_dir, 'data', 'features',
                    'user_profilefeat_lowdim.json'),

            # --- the directory to user activity feature files
            'user_activity_feat_path': os.path.join(
                    root_dir, 'data', 'features',
                    'user_act_features.json'),

            # --- the directory to user cluster
            'user_cluster_path': os.path.join(
                    root_dir, 'data',
                    'normalize_100_user_clusters.json'),
            # low_dim_100_normalize_user_clusters.json
            # normalize_100_user_clusters.json

            # --- the directory to cluster profile feature files
            'cluster_profile_feat_path': os.path.join(
                    root_dir, 'data',
                    'cluster_profile_feat_log_normalization_' +
                    '2015-01-01_2017-07-31',
                    'cluster_profile_feat.pickle'),

            # --- the directory to cluster activity feature files
            'cluster_activity_feat_path': os.path.join(
                    root_dir, 'data',
                    # 'cluster_activity_feat_log10_xplus1_normalization_' +
                    # 'cluster_activity_feat_aug9_' +
                    # '10log10_xplus1_normalization_' +
                    'cluster_activity_feat_log_normalization_' +
                    '2015-01-01_2017-07-31',
                    'cluster_act_features.pickle'),

            # --- the directory to github event repo list
            'github_event_repos_path': os.path.join(
                    root_dir, 'data', 'event_repo_list.pickle'),

            # --- the directory to unique repo list
            'load_repo_path': os.path.join(
                    root_dir, 'data', 'load_repo_train_vali.pickle'),

            # --- the directory to unique_repo_train_vali
            'unique_repo_train_vali_path': '/media/data1/github_paper_exp/exp/www_branch_et/dataset/unique_repo_train_vali.json',

            'root_dir': root_dir
            }
    return path_config


def create_dataset_config():
    dataset_config = {
            'use_repo_embedding': False,
            'use_repo_profile_features': False,
            'use_repo_idx_features': False,
            'use_repo_activity_features': False,
            'use_user_profile_features': False,
            'use_user_activity_features': False,
            'use_event_type_one_hot': False,
            'use_time_delay_features': False,
            'use_user_cluster_one_hot': False,
            'use_user_cluster_minmax': False,
            'use_cluster_profile_features': False,
            'use_cluster_activity_features': False,

            'dim_repo_embedding': 256,
            'dim_repo_profile_features': 2,
            # {'language': [67], 'user_type': [1, 0]}
            'dim_repo_idx_features': 1,
            'dim_repo_activity_features': 52,
            'dim_user_profile_features': 255,
            'dim_user_activity_features': 28,  # 14 * 2
            'dim_time_delay_features': 1,
            'dim_user_cluster_one_hot': 101,
            'dim_user_cluster_minmax': 1,
            'dim_cluster_profile_features': 8,  # 9
            'dim_cluster_activity_features': 52,  # 14*2 + 12 + 12

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
            'num_epochs': 500,
            'loss_weights': {'event_type_output': 1.0,
                             'time_delay_output': 1.0,
                             'user_cluster_output': 1.0,
                             },

            'num_gpu': 2,
            'multiprocessing_cpu': 'all',

            'et_loss': 'bce',  # {'focalloss', 'bce'}
            'focalloss_gamma': 12,  # [0, 2, 4, 6, 8, 10]
            'focalloss_alpha': 0.8,  # [0.2, 0.4, 0.6, 0.7, 0.8, 1]

            'patience': 100,

            'freeze_control': False,
            'freeze_td': 1,
            'freeze_et': 6,

            'lr': 0.001,
            'decay': 0,
            'amsgrad': False,

            'generator_shuffle': True
            }
    return training_config


def create_testing_config():
    testing_config = {
            'load_model_epoch': 12,  # aug7_2: 3 6 16 18 23

            'only_has_event': False,  # only useful when given_gt is False
            'given_gt': False,

            'keep_pred_max': 5,  # keep prediction threshold 20
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
            'user_cluster_nlg_eval': True,
            'event_type_map_eval': True,
            'user_cluster_map_eval': True,
            'event_type_percentage_eval': True,
            'user_cluster_percentage_eval': False,

            'plot_ts': True,

            'nlgeval_repo_dir': '/media/data1/github_paper_exp/nlg-eval/'

            }
    if 'num_test_chains_too_many_no_event_one_month' in config:
        config['num_test_chains_too_many_no_event_one_month'] = len(
                config['test_chains_too_many_no_event_one_month'])

    return config
