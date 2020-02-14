#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 00:53:31 2019

@author: zhouhonglu
"""
from models.multitask_lstm.utils import set_logger
from models.multitask_lstm.utils import print_and_log


import os
import time
import pickle
import sys
from datetime import datetime as dt
import matplotlib.pyplot as plt
from fastdtw import fastdtw
import numpy as np
# import pdb


def eval(config):
    evaluation_start = time.time()

    # load result pickle
    result_save_path = os.path.join(config['exp_save_dir'],
                                    'test_result-epoch{}'.format(
                                            config['load_model_epoch']))
    with open(os.path.join(result_save_path,
                           'result.pickle'), 'rb') as handle:
        result = pickle.load(handle)
    print('result.pickle loaded!')
    print("result.keys: {}\n\n".format(result.keys()))

    # logger
    logger = set_logger(os.path.join(result_save_path, 'evaluate_' +
                                     dt.now().strftime("%Y-%m-%dT%H-%M-%SZ") +
                                     '.log'))
    print_and_log(logger, "Evaluation over {} simulated chains...".format(
            len(result['chain_name'])))

    # evaluation proceses
    if config['event_type_nlg_eval']:
        event_type_nlg_eval(config, logger, result)

    if config['time_delay_overall_evaluation']:
        time_delay_overall_evaluation(config, logger, result)

    print_and_log(logger, "\n\nEvaluation took {} s".format(
            round(time.time() - evaluation_start), 2))
    return


def event_type_nlg_eval(config, logger, result):
    print_and_log(logger, "====================================")
    print_and_log(logger, "event type average bleu scores:")
    print_and_log(logger, "Please install nlg-eval package!\n"
                  "Reference: https://github.com/Maluuba/nlg-eval")
    print_and_log(logger, "After installing, please change the package "
                  "__init__.py file (contact: honglu.zhou@rutgers.edu).")

    sys.path.append(config['nlgeval_repo_dir'])
    from nlgeval import compute_individual_metrics

    # avg  bleu
    avg_bleu = dict()
    for i in range(len(result['chain_name'])):
        gt_chain = " ".join(result['gt_all_event_type'][i])
        hy_chain = " ".join(result['pred_all_event_type'][i])
        metrics_dict = compute_individual_metrics(gt_chain, hy_chain,
                                                  no_overlap=(False, True),
                                                  no_skipthoughts=True,
                                                  no_glove=True)
        for metric in metrics_dict:
            try:
                avg_bleu[metric] += metrics_dict[metric]
            except KeyError:
                avg_bleu[metric] = metrics_dict[metric]

    for metric in avg_bleu:
        avg_bleu[metric] = avg_bleu[metric] / len(result['chain_name'])

    for metric in avg_bleu:
        print_and_log(logger, "{}: {}".format(
                metric, round(avg_bleu[metric], 2)))
    return


def time_delay_overall_evaluation(config, logger, result,
                                  result_save_path=None):
    print_and_log(logger, "====================================")
    print_and_log(logger, "time delay overall evaluation:")

    # statistics
    pred_all = []
    gt_all = []
    avg_dtw = []
    for i in range(len(result['chain_name'])):
        pred_time_delay = result['pred_all_time_delay'][i]
        gt_time_delay = result['gt_all_time_delay'][i]
        avg_dtw.append(fastdtw(gt_time_delay, pred_time_delay)[0])
        pred_all += pred_time_delay
        gt_all += gt_time_delay

    print_and_log(logger, "DTW: {}".format(np.mean(avg_dtw)))
    print_and_log(logger, "MAX predicted: {}, ground truth: {}".format(
                          round(max(pred_all), 2),
                          round(max(gt_all), 2)))
    print_and_log(logger, "MIN predicted: {}, ground truth: {}".format(
                          round(min(pred_all), 2),
                          round(min(gt_all), 2)))
    print_and_log(logger, "MEAN predicted: {}, ground truth: {}".format(
                          round(np.mean(pred_all), 2),
                          round(np.mean(gt_all), 2)))
    print_and_log(logger, "STD predicted: {}, ground truth: {}".format(
                          round(np.std(pred_all), 2),
                          round(np.std(gt_all), 2)))

    # chain length evaluation
    length_stat = dict()
    length_stat["gt_chain_1"] = 0
    length_stat["No_of_chains_diff"] = 0
    length_stat["Same_as_gt"] = 0
    length_stat["diff_1_to_5"] = 0
    length_stat["diff_5_to_10"] = 0
    length_stat["diff_10_to_50"] = 0
    length_stat["diff_50_to_100"] = 0
    length_stat["diff_100+"] = 0
    length_stat["applied_threshold"] = \
        len(result["chains_applied_keep_pred"])

    sim_start = config['sim_period']['start'].split('T')[0]
    sim_end = config['sim_period']['end'].split('T')[0]
    if result_save_path is None:
        result_save_path = os.path.join(config['exp_save_dir'],
                                        'test_result-epoch{}'.format(
                                                config['load_model_epoch']))
    time_delay_plot_save_path = os.path.join(
            result_save_path, "time_delay_plot")

    if not os.path.exists(time_delay_plot_save_path):
        os.makedirs(time_delay_plot_save_path)
    for i in range(len(result['chain_name'])):
        chain = result['chain_name'][i]
        pred_time_delay = result['pred_all_time_delay'][i]
        gt_time_delay = result['gt_all_time_delay'][i]

        plot_time_delay_ts_for_one_chain(chain,
                                         time_delay_plot_save_path,
                                         pred_time_delay,
                                         gt_time_delay,
                                         sim_start, sim_end)
        if len(gt_time_delay) == 1:
            length_stat["gt_chain_1"] += 1
        if len(set(pred_time_delay)) > 1 and len(gt_time_delay) != 1:
            length_stat["No_of_chains_diff"] += 1
        if len(pred_time_delay) == len(gt_time_delay):
            length_stat["Same_as_gt"] += 1
        if abs(len(pred_time_delay) - len(gt_time_delay)) < 5 and (
                abs(len(pred_time_delay) - len(gt_time_delay)) >= 1):
            length_stat["diff_1_to_5"] += 1
        if abs(len(pred_time_delay) - len(gt_time_delay)) < 10 and (
                abs(len(pred_time_delay) - len(gt_time_delay)) >= 5):
            length_stat["diff_5_to_10"] += 1
        if abs(len(pred_time_delay) - len(gt_time_delay)) < 50 and (
                abs(len(pred_time_delay) - len(gt_time_delay)) >= 10):
            length_stat["diff_10_to_50"] += 1
        if abs(len(pred_time_delay) - len(gt_time_delay)) < 100 and (
                abs(len(pred_time_delay) - len(gt_time_delay)) >= 50):
            length_stat["diff_50_to_100"] += 1
        if abs(len(pred_time_delay) - len(gt_time_delay)) >= 100:
            length_stat["diff_100+"] += 1

    print_and_log(logger, "Count of number of simulated chains: {}".format(
                  len(result['chain_name'])))
    print_and_log(logger, "Count of number of predicted chains that has "
                  "same length as ground truth"
                  ": {}, percentage: {}".format(
                        length_stat["Same_as_gt"],
                        round(length_stat["Same_as_gt"]/len(
                                result['chain_name']), 2)))
    print_and_log(logger, "Count of number of predicted chains that "
                  "length difference is 1 to 5: {}, percentage: {}".format(
                          length_stat["diff_1_to_5"],
                          round(length_stat["diff_1_to_5"]/len(
                                  result['chain_name']), 2)))
    print_and_log(logger, "Count of number of predicted chains that "
                  "length difference is 5 to 10: {}, percentage: {}".format(
                          length_stat["diff_5_to_10"],
                          round(length_stat["diff_5_to_10"]/len(
                                  result['chain_name']), 2)))
    print_and_log(logger, "Count of number of predicted chains that "
                  "length difference is 10 to 50: {}, percentage: {}".format(
                          length_stat["diff_10_to_50"],
                          round(length_stat["diff_10_to_50"]/len(
                                  result['chain_name']), 2)))
    print_and_log(logger, "Count of number of predicted chains that "
                  "length difference is 50 to 100: {}, percentage: {}".format(
                          length_stat["diff_50_to_100"],
                          round(length_stat["diff_50_to_100"]/len(
                                  result['chain_name']), 2)))
    print_and_log(logger, "Count of number of predicted chains that "
                  "length difference is 100 and above: {}, "
                  "percentage: {}".format(
                          length_stat["diff_100+"],
                          round(length_stat["diff_100+"]/len(
                                  result['chain_name']), 2)))
    print_and_log(logger, "Count of number of predicted chains that "
                  "length needed threshold to be applied: {}, "
                  "percentage: {} ".format(
                          length_stat["applied_threshold"],
                          round(length_stat["applied_threshold"]/len(
                                  result['chain_name']), 2)))

    print_and_log(logger, "\nTime delay time series plot of each chain "
                  "was plotted in {}/".format(time_delay_plot_save_path))

    return


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
