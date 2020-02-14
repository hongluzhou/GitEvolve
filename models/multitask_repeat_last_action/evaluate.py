#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 00:53:31 2019

@author: zhouhonglu
"""
from models.multitask_repeat_last_action.utils import set_logger
from models.multitask_repeat_last_action.utils import print_and_log


import os
import pdb
import time
import pickle
import sys
from datetime import datetime as dt
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import numpy as np
import ml_metrics
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
# import pdb


def eval(config):
    evaluation_start = time.time()

    if not config['given_gt']:
        result_save_path = os.path.join(
                config['exp_save_dir'],
                'test_result')
        result_evaluation(config, result_save_path, config['only_has_event'])

    else:
        result_save_path = os.path.join(
                config['exp_save_dir'],
                'test_given_gt_result-epoch{}'.format(
                        config['load_model_epoch']))
        result_evaluation_given_gt(config, result_save_path)

    print("\n\nEvaluation took {} s".format(
            round(time.time() - evaluation_start), 2))
    return


def result_evaluation_given_gt(config, result_save_path,
                               only_has_event=False):
    # load result pickle
    with open(os.path.join(result_save_path,
                           'result.pickle'), 'rb') as handle:
        result = pickle.load(handle)
    print('result.pickle loaded!')
    print("result.keys: {}\n\n".format(result.keys()))

    if only_has_event:
        result_new = dict()
        result_new['chain_name'] = list()
        result_new['pred_all_event_id'] = list()
        result_new['pred_all_event_type'] = list()
        result_new['pred_all_time_delay'] = list()
        result_new['pred_all_user_cluster'] = list()
        result_new['gt_all_event_id'] = list()
        result_new['gt_all_event_type'] = list()
        result_new['gt_all_time_delay'] = list()
        result_new['gt_all_user_cluster'] = list()

        for i in range(len(result['chain_name'])):
            if len(result['gt_all_event_id'][i]) != 0:
                result_new['chain_name'].append(
                        result['chain_name'][i]
                        )
                result_new['pred_all_event_id'].append(
                        result['pred_all_event_id'][i]
                        )
                result_new['pred_all_event_type'].append(
                        result['pred_all_event_type'][i]
                        )
                result_new['pred_all_time_delay'].append(
                        result['pred_all_time_delay'][i]
                        )
                result_new['pred_all_user_cluster'].append(
                        result['pred_all_user_cluster'][i]
                        )
                result_new['gt_all_event_id'].append(
                        result['gt_all_event_id'][i]
                        )
                result_new['gt_all_event_type'].append(
                        result['gt_all_event_type'][i]
                        )
                result_new['gt_all_time_delay'].append(
                        result['gt_all_time_delay'][i]
                        )
                result_new['gt_all_user_cluster'].append(
                        result['gt_all_user_cluster'][i]
                        )

        result = result_new

    # logger
    if only_has_event:
        logger = set_logger(os.path.join(
                result_save_path, 'evaluate_only_has_event_given_gt_' +
                dt.now().strftime("%Y-%m-%dT%H-%M-%SZ") + '.log'))
    else:
        logger = set_logger(os.path.join(
                result_save_path, 'evaluate_all_given_gt_' +
                dt.now().strftime("%Y-%m-%dT%H-%M-%SZ") + '.log'))

    print_and_log(logger, "Evaluation over {} simulated chains...".format(
            len(result['chain_name'])))

    # evaluation proceses
    if config['event_type_nlg_eval']:
        result = event_type_nlg_eval(config, logger, result)

    if config['event_type_map_eval']:
        result = event_type_map_eval_given_gt(config, logger, result)

    result = event_type_categorical_accuracy_eval_given_gt(
            config, logger, result)

    if config['event_type_percentage_eval']:
        result = event_type_percentage_eval(config, logger, result)

    if config['user_cluster_nlg_eval']:
        result = user_cluster_nlg_eval(config, logger, result)

    if config['user_cluster_map_eval']:
        result = user_cluster_map_eval_given_gt(config, logger, result)

    result = user_cluster_categorical_accuracy_eval_given_gt(
            config, logger, result)

    if config['user_cluster_percentage_eval']:
        result = user_cluster_percentage_eval(config, logger, result)

    if config['time_delay_overall_evaluation']:
        if not only_has_event:
            result = time_delay_overall_evaluation(
                    config, logger, result, result_save_path,
                    plot_ts=config['plot_ts'], chain_length_eval=False)
        else:
            result = time_delay_overall_evaluation(
                    config, logger, result, result_save_path,
                    plot_ts=False, chain_length_eval=False)

    write_result_to_file(config, result, logger)

    del logger

    return


def result_evaluation(config, result_save_path, only_has_event=False):
    # load result pickle
    with open(os.path.join(result_save_path,
                           'result.pickle'), 'rb') as handle:
        result = pickle.load(handle)
    print('result.pickle loaded!')
    print("result.keys: {}\n\n".format(result.keys()))

    if only_has_event:
        result_new = dict()
        result_new['chain_name'] = list()
        result_new['pred_all_event_id'] = list()
        result_new['pred_all_event_type'] = list()
        result_new['pred_all_time_delay'] = list()
        result_new['pred_all_user_cluster'] = list()
        result_new['gt_all_event_id'] = list()
        result_new['gt_all_event_type'] = list()
        result_new['gt_all_time_delay'] = list()
        result_new['gt_all_user_cluster'] = list()

        for i in range(len(result['chain_name'])):
            if len(result['gt_all_event_id'][i]) != 0:
                result_new['chain_name'].append(
                        result['chain_name'][i]
                        )
                result_new['pred_all_event_id'].append(
                        result['pred_all_event_id'][i]
                        )
                result_new['pred_all_event_type'].append(
                        result['pred_all_event_type'][i]
                        )
                result_new['pred_all_time_delay'].append(
                        result['pred_all_time_delay'][i]
                        )
                result_new['pred_all_user_cluster'].append(
                        result['pred_all_user_cluster'][i]
                        )
                result_new['gt_all_event_id'].append(
                        result['gt_all_event_id'][i]
                        )
                result_new['gt_all_event_type'].append(
                        result['gt_all_event_type'][i]
                        )
                result_new['gt_all_time_delay'].append(
                        result['gt_all_time_delay'][i]
                        )
                result_new['gt_all_user_cluster'].append(
                        result['gt_all_user_cluster'][i]
                        )

        result = result_new

    # logger
    if only_has_event:
        logger = set_logger(os.path.join(
                result_save_path, 'evaluate_only_has_event_' +
                dt.now().strftime("%Y-%m-%dT%H-%M-%SZ") + '.log'))
    else:
        logger = set_logger(os.path.join(
                result_save_path, 'evaluate_all_' +
                dt.now().strftime("%Y-%m-%dT%H-%M-%SZ") + '.log'))

    # evaluation proceses
    if config['event_type_nlg_eval']:
        result = event_type_nlg_eval(config, logger, result)

    if config['event_type_map_eval']:
        result = event_type_map_eval(config, logger, result)

    if config['event_type_percentage_eval']:
        result = event_type_percentage_eval(config, logger, result)

    if config['user_cluster_nlg_eval']:
        result = user_cluster_nlg_eval(config, logger, result)

    if config['user_cluster_map_eval']:
        result = user_cluster_map_eval(config, logger, result)

    if config['user_cluster_percentage_eval']:
        result = user_cluster_percentage_eval(config, logger, result)

    if config['time_delay_overall_evaluation']:
        if not only_has_event:
            result = time_delay_overall_evaluation(
                    config, logger, result, result_save_path,
                    plot_ts=config['plot_ts'], chain_length_eval=True)
        else:
            result = time_delay_overall_evaluation(
                    config, logger, result, result_save_path,
                    plot_ts=False, chain_length_eval=True)

    write_result_to_file(config, result, logger)

    del logger

    return


def write_result_to_file(config, result, logger):

    logger.info("\n\n\nevaluation results of each chain:")

    for i in range(len(result['chain_name'])):
        # logger.info("=====================================================")
        logger.info("testing {}/{}".format(i+1, len(result['chain_name'])))
        logger.info("repo: {}".format(result['chain_name'][i]))
        logger.info("ground truth chain length: {}".format(
                len(result['gt_all_event_id'][i])
                ))
        logger.info("predicted chain length: {}".format(
                len(result['pred_all_event_id'][i])
                ))
        logger.info("et bleu1: {}".format(
                round(result['et_bleu1'][i], 4)
                ))
        if result['et_bleu2'][i] != 'null':
            logger.info("et bleu2: {}".format(
                    round(result['et_bleu2'][i], 4)
                    ))
        if result['et_bleu3'][i] != 'null':
            logger.info("et bleu3: {}".format(
                    round(result['et_bleu3'][i], 4)
                    ))
        if result['et_bleu4'][i] != 'null':
            logger.info("et bleu4: {}".format(
                    round(result['et_bleu4'][i], 4)
                    ))
        logger.info("et AP: {}".format(
                    round(result['et_ap'][i], 4)
                    ))
        logger.info("td DTW: {}".format(
                round(result['td_DTW'][i], 4)
                ))
        if result['td_MSE'][i] != 'null':
            logger.info("td MSE: {}".format(
                    round(result['td_MSE'][i], 4)
                    ))
        logger.info("uc bleu1: {}".format(
                round(result['uc_bleu1'][i], 4)
                ))
        if result['uc_bleu2'][i] != 'null':
            logger.info("uc bleu2: {}".format(
                    round(result['uc_bleu2'][i], 4)
                    ))
        if result['uc_bleu3'][i] != 'null':
            logger.info("uc bleu3: {}".format(
                    round(result['uc_bleu3'][i], 4)
                    ))
        if result['uc_bleu4'][i] != 'null':
            logger.info("uc bleu4: {}".format(
                    round(result['uc_bleu4'][i], 4)
                    ))
        logger.info("uc AP: {}".format(
                    round(result['uc_ap'][i], 4)
                    ))
        logger.info("actual events: {}".format(
                result['gt_all_event_type'][i]
                ))
        logger.info("predicted events: {}".format(
                result['pred_all_event_type'][i]
                ))
        logger.info("actual time delay: {}".format(
                result['gt_all_time_delay'][i]
                ))
        logger.info("predicted time delay: {}".format(
                result['pred_all_time_delay'][i]
                ))
        logger.info("actual user cluster: {}".format(
                result['gt_all_user_cluster'][i]
                ))
        logger.info("predicted user cluster: {}".format(
                result['pred_all_user_cluster'][i]
                ))
        logger.info(" ")
    return


def event_type_categorical_accuracy_eval_given_gt(
                config, logger, result):
    print_and_log(logger, "====================================")
    print_and_log(logger, "event type categorical accuracy evaluation:")

    y_true = []
    y_pred = []

    for a, p in zip(result['gt_all_event_id'], result['pred_all_event_id']):
        y_true += a
        y_pred += p

    result['et_cate'] = accuracy_score(y_true, y_pred)

    print_and_log(logger, "event type categorical accuracy: {}".format(
            round(result['et_cate'], 4)))

    return result


def user_cluster_categorical_accuracy_eval_given_gt(
                config, logger, result):
    print_and_log(logger, "====================================")
    print_and_log(logger, "user cluster categorical accuracy evaluation:")

    y_true = []
    y_pred = []

    for a, p in zip(result['gt_all_user_cluster'],
                    result['pred_all_user_cluster']):
        y_true += a
        y_pred += p

    result['uc_cate'] = accuracy_score(y_true, y_pred)

    print_and_log(logger, "user cluster categorical accuracy: {}".format(
            round(result['uc_cate'], 4)))

    return result


def event_type_map_eval_given_gt(config, logger, result):
    print_and_log(logger, "====================================")
    print_and_log(logger, "event type MAP evaluation:")

    result['et_ap'] = list()

    for a, p in zip(result['gt_all_event_id'], result['pred_all_event_id']):
        AP = precision_score(a, p, average='macro')
        result['et_ap'].append(AP)

    map_re = np.mean(result['et_ap'])
    result['et_map'] = map_re

    print_and_log(logger, "event type MAP: {}".format(round(map_re, 4)))

    return result


def user_cluster_map_eval_given_gt(config, logger, result):
    print_and_log(logger, "====================================")
    print_and_log(logger, "user cluster MAP evaluation:")

    result['uc_ap'] = list()

    for a, p in zip(result['gt_all_user_cluster'],
                    result['pred_all_user_cluster']):
        AP = precision_score(a, p, average='macro')
        result['uc_ap'].append(AP)

    map_re = np.mean(result['uc_ap'])
    result['uc_map'] = map_re

    print_and_log(logger, "user cluster MAP: {}".format(round(map_re, 4)))

    return result


def cal_distribution(counts_per_class):
    total_counts = sum(counts_per_class)
    try:
        distribution = [c/total_counts for c in counts_per_class]
    except ZeroDivisionError:
        pdb.set_trace()
    return distribution


def event_type_percentage_eval(config, logger, result):
    print_and_log(logger, "====================================")
    print_and_log(logger, "event type distribution evaluation:")

    gt_class_list = []
    pred_class_list = []
    for i in range(len(result['chain_name'])):
        gt_class_list += result['gt_all_event_type'][i]
        pred_class_list += result['pred_all_event_type'][i]

    gt_class_list_counter = Counter(gt_class_list)
    pred_class_list_counter = Counter(pred_class_list)

    eventtype_2_id = dict()
    for key in config['eventtype_2_id']:
        eventtype_2_id[key] = config['eventtype_2_id'][key]-1
    id_2_eventtype = dict(zip(eventtype_2_id.values(),
                              eventtype_2_id.keys()))
    # pdb.set_trace()

    counts_per_class = []
    for i in range(len(id_2_eventtype)):
        et = id_2_eventtype[i]
        counts_per_class.append(
                gt_class_list_counter[et])
    gt_distribution = cal_distribution(counts_per_class)

    counts_per_class = []
    for i in range(len(id_2_eventtype)):
        et = id_2_eventtype[i]
        counts_per_class.append(
                pred_class_list_counter[et])
    pred_distribution = cal_distribution(counts_per_class)

    print_and_log(logger, "!!!!  ground truth distribution: ")
    for i in range(len(id_2_eventtype)):
        et = id_2_eventtype[i]
        print_and_log(logger, "{}: {}".format(
                et, round(gt_distribution[i], 4)))

    print_and_log(logger, "!!!!  prediction distribution: ")
    for i in range(len(id_2_eventtype)):
        et = id_2_eventtype[i]
        print_and_log(logger, "{}: {}".format(
                et, round(pred_distribution[i], 4)))
    return result


def user_cluster_percentage_eval(config, logger, result):
    print_and_log(logger, "====================================")
    print_and_log(logger, "user cluster distribution evaluation:")

    gt_class_list = []
    pred_class_list = []
    for i in range(len(result['chain_name'])):
        gt_class_list += result['gt_all_user_cluster'][i]
        pred_class_list += result['pred_all_user_cluster'][i]

    gt_class_list_counter = Counter(gt_class_list)
    pred_class_list_counter = Counter(pred_class_list)

    clusters = list(range(100 + 1))

    counts_per_class = []
    for i in range(len(clusters)):
        counts_per_class.append(
                gt_class_list_counter[i])
    gt_distribution = cal_distribution(counts_per_class)

    counts_per_class = []
    for i in range(len(clusters)):
        counts_per_class.append(
                pred_class_list_counter[i])
    pred_distribution = cal_distribution(counts_per_class)

    print_and_log(logger, "!!!!  ground truth distribution: ")
    for i in range(len(clusters)):
        print_and_log(logger, "{}: {}".format(
                i, round(gt_distribution[i], 4)))

    print_and_log(logger, "!!!!  prediction distribution: ")
    for i in range(len(clusters)):
        print_and_log(logger, "{}: {}".format(
                i, round(pred_distribution[i], 4)))
    return result


def compute_AP(actual, predicted):
    k = len(actual)

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def compute_MAP(actual, predicted):
    return np.mean([compute_AP(a, p) for a, p in zip(actual, predicted)])


def event_type_map_eval(config, logger, result):
    print_and_log(logger, "====================================")
    print_and_log(logger, "event type MAP evaluation:")

    result['et_ap'] = list()

    for a, p in zip(result['gt_all_event_id'], result['pred_all_event_id']):
        AP = compute_AP(a, p)
        result['et_ap'].append(AP)

    map_re = np.mean(result['et_ap'])
    result['et_map'] = map_re

    print_and_log(logger, "event type MAP: {}".format(round(map_re, 4)))

    return result


def event_type_map_eval_ml_metrics(config, logger, result):
    print_and_log(logger, "====================================")
    print_and_log(logger, "event type MAP evaluation:")

    print_and_log(logger, "event type MAP: {}".format(
            ml_metrics.mapk(result['gt_all_event_id'],
                            result['pred_all_event_id'])
            ))

    k_list = []
    for i in range(len(result['chain_name'])):
        k_list.append(len(result['gt_all_event_type'][i]))
    k_list = sorted(list(set(k_list)))
    k_list.remove(0)
    print_and_log(logger, "all possible k: {}".format(k_list))

    for k in k_list:
        map_at_k = ml_metrics.mapk(result['gt_all_event_id'],
                                   result['pred_all_event_id'],
                                   k)
        print_and_log(logger, "event type MAP@{}: {}".format(
                int(k), map_at_k))

    return result


def user_cluster_map_eval(config, logger, result):
    print_and_log(logger, "====================================")
    print_and_log(logger, "user cluster MAP evaluation:")

    result['uc_ap'] = list()

    for a, p in zip(result['gt_all_user_cluster'],
                    result['pred_all_user_cluster']):
        AP = compute_AP(a, p)
        result['uc_ap'].append(AP)

    map_re = np.mean(result['uc_ap'])
    result['uc_map'] = map_re

    print_and_log(logger, "user cluster MAP: {}".format(round(map_re, 4)))

    return result


def user_cluster_map_eval_ml_metrics(config, logger, result):
    print_and_log(logger, "====================================")
    print_and_log(logger, "user cluster MAP evaluation:")

    print_and_log(logger, "user cluster MAP: {}".format(
            ml_metrics.mapk(result['gt_all_user_cluster'],
                            result['pred_all_user_cluster'])
            ))

    k_list = []
    for i in range(len(result['chain_name'])):
        k_list.append(len(result['gt_all_user_cluster'][i]))
    k_list = sorted(list(set(k_list)))
    k_list.remove(0)
    print_and_log(logger, "all possible k: {}".format(k_list))

    for k in k_list:
        map_at_k = ml_metrics.mapk(result['gt_all_user_cluster'],
                                   result['pred_all_user_cluster'],
                                   k)
        print_and_log(logger, "user cluster MAP@{}: {}".format(
                int(k), map_at_k))

    return result


def event_type_nlg_eval(config, logger, result):
    print_and_log(logger, "====================================")
    print_and_log(logger, "event type average bleu scores:")
    # print_and_log(logger, "Please install nlg-eval package!\n"
    #               "Reference: https://github.com/Maluuba/nlg-eval")
    # print_and_log(logger, "After installing, please change the package "
    #               "__init__.py file (contact: honglu.zhou@rutgers.edu).")

    sys.path.append(config['nlgeval_repo_dir'])
    from nlgeval import compute_individual_metrics

    # avg  bleu
    avg_bleu = dict()
    avg_bleu['Bleu_1'] = list()
    avg_bleu['Bleu_2'] = list()
    avg_bleu['Bleu_3'] = list()
    avg_bleu['Bleu_4'] = list()

    result['et_bleu1'] = list()
    result['et_bleu2'] = list()
    result['et_bleu3'] = list()
    result['et_bleu4'] = list()
    for i in range(len(result['chain_name'])):
        if len(result['gt_all_event_type'][i]) == 0:
            gt_chain = " ".join(['no_event_in_simperiod'])
        else:
            gt_chain = " ".join(result['gt_all_event_type'][i])

        if len(result['pred_all_event_type'][i]) == 0:
            hy_chain = " ".join(['no_event_in_simperiod'])
        else:
            hy_chain = " ".join(result['pred_all_event_type'][i])
        metrics_dict = compute_individual_metrics(gt_chain, hy_chain,
                                                  no_overlap=(False, True),
                                                  no_skipthoughts=True,
                                                  no_glove=True)

        result['et_bleu1'].append(metrics_dict['Bleu_1'])
        avg_bleu['Bleu_1'].append(metrics_dict['Bleu_1'])

        # cond = True

        # if cond:
        if len(result['gt_all_event_type'][i]) >= 2:  # and (
                # len(result['pred_all_event_type'][i]) >= 2
                # ):
            result['et_bleu2'].append(metrics_dict['Bleu_2'])
            avg_bleu['Bleu_2'].append(metrics_dict['Bleu_2'])
        else:
            result['et_bleu2'].append('null')

        # if cond:
        if len(result['gt_all_event_type'][i]) >= 3:  # and (
                # len(result['pred_all_event_type'][i]) >= 3
                # ):
            result['et_bleu3'].append(metrics_dict['Bleu_3'])
            avg_bleu['Bleu_3'].append(metrics_dict['Bleu_3'])
        else:
            result['et_bleu3'].append('null')

        # if cond:
        if len(result['gt_all_event_type'][i]) >= 4:  # and (
                # len(result['pred_all_event_type'][i]) >= 4
                # ):
            result['et_bleu4'].append(metrics_dict['Bleu_4'])
            avg_bleu['Bleu_4'].append(metrics_dict['Bleu_4'])
        else:
            result['et_bleu4'].append('null')

    for metric in avg_bleu:
        print_and_log(logger, "{}: {}".format(
                metric, round(np.average(avg_bleu[metric]), 4)))
#        print_and_log(logger, "{}: {}, calculated from {} values".format(
#                metric, round(np.average(avg_bleu[metric]), 4),
#                len(avg_bleu[metric])))
    # pdb.set_trace()
    return result


def user_cluster_nlg_eval(config, logger, result):
    print_and_log(logger, "====================================")
    print_and_log(logger, "user cluster average bleu scores:")
    # print_and_log(logger, "Please install nlg-eval package!\n"
    #               "Reference: https://github.com/Maluuba/nlg-eval")
    # print_and_log(logger, "After installing, please change the package "
    #               "__init__.py file (contact: honglu.zhou@rutgers.edu).")

    sys.path.append(config['nlgeval_repo_dir'])
    from nlgeval import compute_individual_metrics

    # avg  bleu
    avg_bleu = dict()
    avg_bleu = dict()
    avg_bleu['Bleu_1'] = list()
    avg_bleu['Bleu_2'] = list()
    avg_bleu['Bleu_3'] = list()
    avg_bleu['Bleu_4'] = list()

    result['uc_bleu1'] = list()
    result['uc_bleu2'] = list()
    result['uc_bleu3'] = list()
    result['uc_bleu4'] = list()
    for i in range(len(result['chain_name'])):
        if len(result['gt_all_user_cluster'][i]) == 0:
            gt_chain = " ".join(['no_event_in_simperiod'])
        else:
            gt_chain = " ".join(
                    [str(ele) for ele in result['gt_all_user_cluster'][i]])
        if len(result['pred_all_user_cluster'][i]) == 0:
            hy_chain = " ".join(['no_event_in_simperiod'])
        else:
            hy_chain = " ".join(
                    [str(ele) for ele in result['pred_all_user_cluster'][i]])
        metrics_dict = compute_individual_metrics(gt_chain, hy_chain,
                                                  no_overlap=(False, True),
                                                  no_skipthoughts=True,
                                                  no_glove=True)
        result['uc_bleu1'].append(metrics_dict['Bleu_1'])
        avg_bleu['Bleu_1'].append(metrics_dict['Bleu_1'])

        if len(result['gt_all_user_cluster'][i]) >= 2:  # and (
                # len(result['pred_all_user_cluster'][i]) >= 2
                # ):
            result['uc_bleu2'].append(metrics_dict['Bleu_2'])
            avg_bleu['Bleu_2'].append(metrics_dict['Bleu_2'])
        else:
            result['uc_bleu2'].append('null')

        if len(result['gt_all_user_cluster'][i]) >= 3:  # and (
                # len(result['pred_all_user_cluster'][i])
                # ):
            result['uc_bleu3'].append(metrics_dict['Bleu_3'])
            avg_bleu['Bleu_3'].append(metrics_dict['Bleu_3'])
        else:
            result['uc_bleu3'].append('null')

        if len(result['gt_all_user_cluster'][i]) >= 4:  # and (
                # len(result['pred_all_user_cluster'][i]) >= 4
                # ):
            result['uc_bleu4'].append(metrics_dict['Bleu_4'])
            avg_bleu['Bleu_4'].append(metrics_dict['Bleu_4'])
        else:
            result['uc_bleu4'].append('null')

    for metric in avg_bleu:
        print_and_log(logger, "{}: {}".format(
                metric, round(np.average(avg_bleu[metric]), 4)))
#        print_and_log(logger, "{}: {}, calculated from {} values".format(
#                metric, round(np.average(avg_bleu[metric]), 4),
#                len(avg_bleu[metric])))
    # pdb.set_trace()
    return result


def time_delay_overall_evaluation(config, logger, result, result_save_path,
                                  plot_ts=True, chain_length_eval=True):
    print_and_log(logger, "====================================")
    print_and_log(logger, "time delay evaluation:")

    # statistics
    pred_all = []
    gt_all = []
    avg_dtw = []
    avg_mse = []
    result["td_DTW"] = list()
    result["td_MSE"] = list()
    for i in range(len(result['chain_name'])):
        pred_time_delay = result['pred_all_time_delay'][i]
        gt_time_delay = result['gt_all_time_delay'][i]
        if len(pred_time_delay) == 0:
            pred_time_delay = [-1]
        if len(gt_time_delay) == 0:
            gt_time_delay = [-1]
        avg_dtw.append(fastdtw(gt_time_delay, pred_time_delay)[0])
        result["td_DTW"].append(avg_dtw[-1])
        if len(gt_time_delay) == len(pred_time_delay):
            avg_mse.append(mean_squared_error(gt_time_delay, pred_time_delay))
            result["td_MSE"].append(avg_mse[-1])
        else:
            result["td_MSE"].append('null')
        if len(result['pred_all_time_delay'][i]) != 0:
            pred_all += pred_time_delay
        if len(result['gt_all_time_delay'][i]) != 0:
            gt_all += gt_time_delay

    print_and_log(logger, "Average DTW: {}".format(round(np.mean(avg_dtw), 4)))
    if config['given_gt']:
        print_and_log(logger, "Average MSE: {}".format(np.mean(avg_mse)))
    print_and_log(logger, "MAX predicted: {}, ground truth: {}".format(
                          round(max(pred_all), 4),
                          round(max(gt_all), 4)))
    print_and_log(logger, "MIN predicted: {}, ground truth: {}".format(
                          round(min(pred_all), 4),
                          round(min(gt_all), 4)))
    print_and_log(logger, "MEAN predicted: {}, ground truth: {}".format(
                          round(np.mean(pred_all), 4),
                          round(np.mean(gt_all), 4)))
    print_and_log(logger, "STD predicted: {}, ground truth: {}".format(
                          round(np.std(pred_all), 4),
                          round(np.std(gt_all), 4)))

    # chain length evaluation
    if chain_length_eval:
        length_mae = []
        length_stat = dict()
        length_stat["gt_chain_0"] = 0
        length_stat["gt_chain_1"] = 0
        length_stat["Same_as_gt"] = 0
        length_stat["diff_1_to_10"] = 0
        length_stat["diff_10_to_100"] = 0
        length_stat["diff_100+"] = 0

    if 'chains_applied_keep_pred' in result:
        length_stat["applied_threshold"] = len(
                result["chains_applied_keep_pred"])

    sim_start = config['sim_period']['start'].split('T')[0]
    sim_end = config['sim_period']['end'].split('T')[0]

    if plot_ts:
        time_delay_plot_save_path = os.path.join(
                result_save_path, "time_delay_plot")
        if not os.path.exists(time_delay_plot_save_path):
            os.makedirs(time_delay_plot_save_path)

    if chain_length_eval or plot_ts:
        for i in range(len(result['chain_name'])):
            chain = result['chain_name'][i]
            pred_time_delay = result['pred_all_time_delay'][i]
            gt_time_delay = result['gt_all_time_delay'][i]

            if plot_ts:
                plot_time_delay_ts_for_one_chain(chain,
                                                 time_delay_plot_save_path,
                                                 pred_time_delay,
                                                 gt_time_delay,
                                                 sim_start, sim_end)

            if chain_length_eval:
                length_mae.append(
                        abs(len(pred_time_delay) - len(gt_time_delay)))

                if len(gt_time_delay) == 0:
                    length_stat["gt_chain_0"] += 1
                if len(gt_time_delay) == 1:
                    length_stat["gt_chain_1"] += 1
                if len(pred_time_delay) == len(gt_time_delay):
                    length_stat["Same_as_gt"] += 1
                if abs(len(pred_time_delay) - len(gt_time_delay)) < 10 and (
                        abs(len(pred_time_delay) - len(gt_time_delay)) >= 1):
                    length_stat["diff_1_to_10"] += 1
                if abs(len(pred_time_delay) - len(gt_time_delay)) < 100 and (
                        abs(len(pred_time_delay) - len(gt_time_delay)) >= 10):
                    length_stat["diff_10_to_100"] += 1
                if abs(len(pred_time_delay) - len(gt_time_delay)) >= 100:
                    length_stat["diff_100+"] += 1

    if chain_length_eval:
        length_mae = np.mean(length_mae)

    if chain_length_eval:
        print_and_log(logger, "====================================")
        print_and_log(logger, "chain length evaluation:")

        print_and_log(logger, "MAE: {}".format(round(length_mae, 4)))

        print_and_log(logger, "Count of number of simulated "
                      "chains: {}".format(len(result['chain_name'])))

        print_and_log(logger, "Count of number of chains whose "
                      "ground truth length is 0: {}".format(
                              length_stat["gt_chain_0"]
                              ))

        print_and_log(logger, "Count of number of chains whose "
                      "ground truth length is 1: {}".format(
                              length_stat["gt_chain_1"]
                              ))

        if 'chains_applied_keep_pred' in result:
            print_and_log(logger, "Count of number of predicted chains that "
                          "length needed threshold to be applied: {}, "
                          "percentage: {} ".format(
                                  length_stat["applied_threshold"],
                                  round(length_stat["applied_threshold"]/len(
                                          result['chain_name']), 4)))

        print_and_log(logger, "Count of number of predicted "
                      "chains that has "
                      "same length as ground truth"
                      ": {}, percentage: {}".format(
                            length_stat["Same_as_gt"],
                            round(length_stat["Same_as_gt"]/len(
                                    result['chain_name']), 4)))
        print_and_log(logger, "Count of number of predicted chains that "
                      "length difference is 1 to 10: {},"
                      "percentage: {}".format(
                              length_stat["diff_1_to_10"],
                              round(length_stat["diff_1_to_10"]/len(
                                      result['chain_name']), 4)))
        print_and_log(logger, "Count of number of predicted chains that "
                      "length difference is 10 to 100: {}, "
                      "percentage: {}".format(
                              length_stat["diff_10_to_100"],
                              round(length_stat["diff_10_to_100"]/len(
                                      result['chain_name']), 4)))
        print_and_log(logger, "Count of number of predicted chains that "
                      "length difference is 100 and above: {}, "
                      "percentage: {}".format(
                              length_stat["diff_100+"],
                              round(length_stat["diff_100+"]/len(
                                      result['chain_name']), 4)))

    return result


def plot_time_delay_ts_for_one_chain(chain, save_dir,
                                     pred_time_delay, gt_time_delay,
                                     sim_start, sim_end):
    plt.figure(1)
    plt.plot(gt_time_delay, 'g-', marker='.', label='ground-truth_time_delay')
    plt.plot(pred_time_delay, 'r-', marker='x',
             label='predicted_time_delay')
    plt.ylabel('time delay value')
    plt.xlabel('event')
    plt.legend()
    plt.grid(True)
    plt.title('The time delay time series - \n{}'.format(
            chain))
    plt.savefig(os.path.join(save_dir, '{}_{}-{}.png'.format(
            chain, sim_start, sim_end)))
    plt.close()


def plot_time_delay_ts_for_one_chain_pred(chain, save_dir,
                                          pred_time_delay,
                                          gt_time_delay,
                                          sim_start, sim_end):
    plt.figure(1)
    plt.plot(pred_time_delay[:len(gt_time_delay)], 'r-', marker='x',
             label='predicted_time_delay')
    plt.ylabel('time delay value')
    plt.xlabel('event')
    plt.legend()
    plt.grid(True)
    plt.title('The time delay time series - \n{}'.format(chain))
    plt.savefig(os.path.join(save_dir, '{}_{}-{}.png'.format(
            chain, sim_start, sim_end)))
    plt.close()
