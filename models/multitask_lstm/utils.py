#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:45:05 2019

@author: zhouhonglu
"""
import logging
import os
import numpy as np
import gzip
import pandas as pd
import json
import datetime
import pdb
from datetime import datetime as dt
from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Reference:
        https://github.com/umbertogriffo/focal-loss-keras
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return - K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Reference:
        https://github.com/umbertogriffo/focal-loss-keras
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)
    return categorical_focal_loss_fixed


def set_logger(logger_name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(logger_name, mode='w')
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.info(os.path.basename(__file__))
    logger.info(dt.now().strftime('%m/%d/%Y %I:%M:%S %p'))

    return logger


def print_and_log(logger, msg):
    print(msg)
    logger.info(msg)


def time_delay_normalization(x, time_delay_normalization_func):
    if time_delay_normalization_func == 'log10_xplus1':
        return np.log10(x + 1)
    elif time_delay_normalization_func == '10log10_xplus1':
        return 10 * np.log10(x + 1)


def time_delay_normalization_reverse(x, log_plus_func):
    if log_plus_func == 'log10_xplus1':
        return pow(10, x) - 1
    elif log_plus_func == '10log10_xplus1':
        return pow(10, x/10) - 1


def load_jsongz_2_dataframe(path2file):
    data = []
    with gzip.open(path2file, 'rb') as fin:
        for line in fin:
            data.append(json.loads(line.decode('utf-8')))

    return pd.DataFrame(data)


def insert_no_event_for_a_chain(config, one_chain_event, one_chain_time,
                                one_chain_user=None):
    one_chain_event_new = []
    one_chain_time_new = []
    if one_chain_user is not None:
        one_chain_user_new = []
    for i in range(len(one_chain_time)):
        if one_chain_event[i] == config['empty_event_type'] or (
                one_chain_event[i] == config['eventtype_2_id']['<soc>']):
            one_chain_event_new.append(one_chain_event[i])
            if one_chain_user is not None:
                one_chain_user_new.append(one_chain_user[i])
            one_chain_time_new.append(one_chain_time[i])
        elif one_chain_event[i-1] == config['eventtype_2_id']['<soc>']:
            one_chain_event_new.append(one_chain_event[i])
            if one_chain_user is not None:
                one_chain_user_new.append(one_chain_user[i])
            one_chain_time_new.append(one_chain_time[i])
        else:
            time_delay = get_time_delay(one_chain_time[i-1],
                                        one_chain_time[i],
                                        'float')[1]
            # if time delay is greater than 30 days
            if time_delay > 30*24:
                one_chain_event_new.append(config['eventtype_2_id'][
                        "no_event_for_1month"])
                if one_chain_user is not None:
                    one_chain_user_new.append("no_event_user")
                one_chain_time_new.append(
                        one_chain_time[i-1] + 30.0*24*3600)
                time_delay = get_time_delay(one_chain_time_new[-1],
                                            one_chain_time[i],
                                            'float')[1]
                while time_delay > 30*24:
                    one_chain_event_new.append(config['eventtype_2_id'][
                            "no_event_for_1month"])
                    if one_chain_user is not None:
                        one_chain_user_new.append("no_event_user")
                    one_chain_time_new.append(
                            one_chain_time_new[-1] + 30.0*24*3600)
                    time_delay = get_time_delay(one_chain_time_new[-1],
                                                one_chain_time[i],
                                                'float')[1]
            one_chain_event_new.append(one_chain_event[i])
            if one_chain_user is not None:
                one_chain_user_new.append(one_chain_user[i])
            one_chain_time_new.append(one_chain_time[i])

    if one_chain_user is not None:
        return (one_chain_event_new, one_chain_time_new, one_chain_user_new)
    else:
        return (one_chain_event_new, one_chain_time_new)


def insert_no_event_for_a_sim_GTchain(config,
                                      one_chain_event, one_chain_time,
                                      input_last_time,
                                      one_chain_user=None):
    one_chain_event_new = []
    one_chain_time_new = []
    if one_chain_user is not None:
        one_chain_user_new = []
    time_delay = get_time_delay(input_last_time,
                                one_chain_time[0],
                                'float')[1]
    if time_delay > 30*24:  # time delay is greater than 30 days
        one_chain_event_new.append(config['eventtype_2_id'][
                "no_event_for_1month"])
        if one_chain_user is not None:
            one_chain_user_new.append("no_event_user")
        one_chain_time_new.append(
                input_last_time + 30.0*24*3600)
        time_delay = get_time_delay(one_chain_time_new[-1],
                                    one_chain_time[0],
                                    'float')[1]
        while time_delay > 30*24:
            one_chain_event_new.append(config['eventtype_2_id'][
                "no_event_for_1month"])
            if one_chain_user is not None:
                one_chain_user_new.append("no_event_user")
            one_chain_time_new.append(
                    one_chain_time_new[-1] + 30.0*24*3600)
            time_delay = get_time_delay(one_chain_time_new[-1],
                                        one_chain_time[0],
                                        'float')[1]
    one_chain_event_new.append(one_chain_event[0])
    if one_chain_user is not None:
        one_chain_user_new.append(one_chain_user[0])
    one_chain_time_new.append(one_chain_time[0])

    for i in range(1, len(one_chain_time)):
        time_delay = get_time_delay(one_chain_time_new[-1],
                                    one_chain_time[i],
                                    'float')[1]
        if time_delay > 30*24:  # time delay is greater than 30 days
            one_chain_event_new.append(config['eventtype_2_id'][
                    "no_event_for_1month"])
            if one_chain_user is not None:
                one_chain_user_new.append("no_event_user")
            one_chain_time_new.append(
                    one_chain_time[i-1] + 30.0*24*3600)
            time_delay = get_time_delay(one_chain_time_new[-1],
                                        one_chain_time[i],
                                        'float')[1]
            while time_delay > 30*24:
                one_chain_event_new.append(config['eventtype_2_id'][
                    "no_event_for_1month"])
                if one_chain_user is not None:
                    one_chain_user_new.append("no_event_user")
                one_chain_time_new.append(
                        one_chain_time_new[-1] + 30.0*24*3600)
                time_delay = get_time_delay(one_chain_time_new[-1],
                                            one_chain_time[i],
                                            'float')[1]
        one_chain_event_new.append(one_chain_event[i])
        if one_chain_user is not None:
            one_chain_user_new.append(one_chain_user[i])
        one_chain_time_new.append(one_chain_time[i])

    if one_chain_user is not None:
        return (one_chain_event_new, one_chain_time_new, one_chain_user_new)
    else:
        return (one_chain_event_new, one_chain_time_new)


def get_time_delay(t1, t2, arg):

    timediff = t2 - t1
    if arg == 'combined':
        d = divmod(timediff, 60 * 60 * 24)
        h = divmod(d[1], 60 * 60)
        m = divmod(h[1], 60)
        s = m[1]
        return ([d[0], h[0], m[0], s])
    elif arg == 'separate':
        d = divmod(timediff, 60 * 60 * 24)
        h = divmod(timediff, 60 * 60)
        m = divmod(timediff, 60)
        s = timediff
        return ([d[0], h[0], m[0], s])
    elif arg == 'float':
        d = timediff/(60 * 60 * 24)
        h = timediff/(60 * 60)
        m = timediff/60
        s = timediff
        return (d, h, m, s)
    else:
        raise ValueError('wrong argument')


def utc_timestamp(timestamp):
    time_format = "%Y-%m-%dT%H:%M:%SZ"
    base_timestep = datetime.datetime(1970, 1, 1)
    timestamp = dt.strptime(timestamp, time_format)
    curr_utc = int((timestamp - base_timestep).total_seconds())
    return curr_utc


def preprocess_cascade_files_to_remove_unnesseary_chains(
        config, cascades_path, github_event_repos_set):
    print("preprocess_cascade_files_to_remove_unnesseary_chains started...")
    all_id_list = [f for f in os.listdir(cascades_path) if "id_" in f]

    profile_repos = dict()
    for info_id in all_id_list:
        reoo_list = [f[:-8] for f in os.listdir(os.path.join(
                cascades_path, info_id)) if '.json.gz' in f]
        for repo in reoo_list:
            # remove repos' chain that should be ignore
            if repo in config['repos_to_ignore']:
                os.remove(os.path.join(cascades_path, info_id,
                                       repo + '.json.gz'))

            # remove event repo's chains
            elif repo in github_event_repos_set:
                os.remove(os.path.join(cascades_path, info_id,
                                       repo + '.json.gz'))

            # only keep profile repo's longest chain
            else:
                try:
                    profile_repos[repo].add(info_id)
                except KeyError:
                    profile_repos[repo] = {info_id}

    for repo in profile_repos:
        infos_thisrepo = list(profile_repos[repo])
        chains_length = []
        for info_id in infos_thisrepo:
            one_chain_pd = load_jsongz_2_dataframe(
                    os.path.join(cascades_path, info_id,
                                 repo + '.json.gz'))
            chains_length.append(len(one_chain_pd))
        info2keep = infos_thisrepo[np.argmax(chains_length)]
        for info in infos_thisrepo:
            if info != info2keep:
                # remove profile repo's other chains, only keep the longest
                os.remove(os.path.join(cascades_path,
                                       info,
                                       repo + '.json.gz'))
    print("preprocess_cascade_files_to_remove_unnesseary_chains done!")


def count_profile_repos(
        config, cascades_path, github_event_repos_set):
    print("count profile_repos started...")
    all_id_list = [f for f in os.listdir(cascades_path) if "id_" in f]

    profile_repos = dict()
    for info_id in all_id_list:
        reoo_list = [f[:-8] for f in os.listdir(os.path.join(
                cascades_path, info_id)) if '.json.gz' in f]
        for repo in reoo_list:
            # super repo
            if repo in config['repos_to_ignore']:
                continue

            # event repo
            elif repo in github_event_repos_set:
                continue

            # profile repo
            else:
                try:
                    profile_repos[repo].add(info_id)
                except KeyError:
                    profile_repos[repo] = {info_id}

    print("number of unique profile repos: {}".format(len(profile_repos)))
    # pdb.set_trace()
