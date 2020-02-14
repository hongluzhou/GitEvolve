#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 22:08:50 2019

@author: zhouhonglu
"""
from utils import set_logger
from utils import print_and_log

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime as dt
import pdb


if __name__ == '__main__':

    # ============= load history files
    model_path = '/media/data1/github_paper_exp/exp/myexp/models/'
    history_file = 'history_final.pickle'
    # history_file = 'history-{}.pickle'.format('105')

    with open(os.path.join(model_path, history_file), 'rb') as f:
        history = pickle.load(f)
    print("model history keys: {}".format(list(history.keys())))
    # pdb.set_trace()

    # ============= create dir if not exists
    save_dir = os.path.join(model_path, 'model_history_figures')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = set_logger(os.path.join(save_dir,
                                     'model_history_' +
                                     dt.now().strftime("%Y-%m-%dT%H-%M-%SZ") +
                                     '.log'))

    # ============= loss figures
    # figsize_settitng = (10, 8*4)
    figsize_settitng = (10, 8)
    plt.figure(figsize=figsize_settitng, dpi=150)

    # plt.subplot(4, 1, 1)
    plt.title('overall training, validation losses')
    plt.plot(history['loss'], '-', marker='.',
             label='overall training loss')
    plt.plot(history['val_loss'], '-', marker='.',
             label='overall validation loss')
    plt.xlabel('iteration')
    plt.ylabel('loss value')
    plt.legend()

    # plt.subplot(4, 1, 2)
    # plt.title('event type training, validation loss figure')
    # plt.plot(history['event_type_output_loss'], '-', marker='.',
    #          label='event type training loss')
    # plt.plot(history['val_event_type_output_loss'], '-', marker='.',
    #          label='event type validation loss')
    # plt.xlabel('iteration')
    # plt.ylabel('loss value')
    # plt.legend()

    # plt.subplot(4, 1, 3)
    # plt.title('time delay training, validation loss figure')
    # plt.plot(history['time_delay_output_loss'], '-', marker='.',
    #          label='time delay training loss')
    # plt.plot(history['val_time_delay_output_loss'], '-', marker='.',
    #          label='time delay validation loss')
    # plt.xlabel('iteration')
    # plt.ylabel('loss value')
    # plt.legend()

    # plt.subplot(4, 1, 4)
    # plt.title('user cluster training, validation loss figure')
    # plt.plot(history['user_cluster_output_loss'], '-', marker='.',
    #          label='user cluster training loss')
    # plt.plot(history['val_user_cluster_output_loss'], '-', marker='.',
    #          label='user cluster validation loss')
    # plt.xlabel('iteration')
    # plt.ylabel('loss value')
    # plt.legend()

    plt.savefig(os.path.join(save_dir, '{}.png'.format(
            "losses")), bbox_inches='tight')
    plt.close()

    # ============= accuracy figures
    # figsize_settitng = (10, 8*3)
    figsize_settitng = (10, 8)
    plt.figure(figsize=figsize_settitng, dpi=150)

    # plt.subplot(3, 1, 1)
    plt.title('accuracy')
    plt.plot(history['categorical_accuracy'], '-',
             marker='.',
             label='training categorical accuracy')
    plt.plot(history['val_categorical_accuracy'], '-',
             marker='.',
             label='validation categorical accuracy')
    plt.xlabel('iteration')
    plt.ylabel('accuracy value')
    plt.legend()

    # plt.subplot(3, 1, 2)
    # plt.title('time delay accuracy')
    # plt.plot(history['time_delay_output_mean_squared_error'], '-',
    #          marker='.',
    #          label='time delay training mean squared error')
    # plt.plot(history['val_time_delay_output_mean_squared_error'], '-',
    #          marker='.',
    #          label='time delay validation mean squared error')
    # plt.xlabel('iteration')
    # plt.ylabel('accuracy value')
    # plt.legend()

    # if 'user_cluster_output_categorical_accuracy' in history:
    #     plt.subplot(3, 1, 3)
    #     plt.title('user cluster accuracy')
    #     plt.plot(history['user_cluster_output_categorical_accuracy'], '-',
    #              marker='.',
    #              label='user cluster training categorical accuracy')
    #     plt.plot(history['val_user_cluster_output_categorical_accuracy'], '-',
    #              marker='.',
    #              label='user cluster validation categorical accuracy')
    #     plt.xlabel('iteration')
    #     plt.ylabel('accuracy value')
    #     plt.legend()
    # else:
    #     plt.subplot(3, 1, 3)
    #     plt.title('user cluster accuracy')
    #     plt.plot(history['user_cluster_output_mean_squared_error'], '-',
    #              marker='.',
    #              label='user cluster training mean squared error')
    #     plt.plot(history['val_user_cluster_output_mean_squared_error'], '-',
    #              marker='.',
    #              label='user cluster validation mean squared error')
    #     plt.xlabel('iteration')
    #     plt.ylabel('accuracy value')
    #     plt.legend()

    plt.savefig(os.path.join(save_dir, '{}.png'.format(
            "accuracy_train_vali")), bbox_inches='tight')
    plt.close()

    # ============= print result
    print_and_log(logger, "\n\nTotal epoch: {}".format(len(history['loss'])))

    print_and_log(logger, "======== Train LOSS")

    print_and_log(logger, "Lowest training loss was at epoch {}".format(
            np.argmin(history['loss'])+1))

    print_and_log(logger, "Highest training loss was at epoch {}".format(
            np.argmax(history['loss'])+1))

    # print_and_log(logger, "Lowest event type training loss was at "
    #               "epoch {}".format(np.argmin(
    #                       history['event_type_output_loss'])+1))

    # print_and_log(logger, "Highest event type training loss was at "
    #               "epoch {}".format(np.argmin(
    #                       history['event_type_output_loss'])+1))

    # print_and_log(logger, "Lowest time delay training loss was at "
    #               "epoch {}".format(np.argmin(
    #                       history['time_delay_output_loss'])+1))

    # print_and_log(logger, "Highest time delay training loss was at "
    #               "epoch {}".format(np.argmax(
    #                       history['time_delay_output_loss'])+1))

    # print_and_log(logger, "Lowest user cluster training loss was at "
    #               "epoch {}".format(np.argmin(
    #                       history['user_cluster_output_loss'])+1))

    # print_and_log(logger, "Highest user cluster training loss was at "
    #               "epoch {}".format(np.argmax(
    #                       history['user_cluster_output_loss'])+1))

    print_and_log(logger, "======== Validation LOSS")

    print_and_log(logger, "Lowest validation loss was at epoch {}".format(
            np.argmin(history['val_loss'])+1))

    print_and_log(logger, "Highest validation loss was at epoch {}".format(
            np.argmax(history['val_loss'])+1))

    # print_and_log(logger, "Lowest event type validation loss was at "
    #               "epoch {}".format(np.argmin(
    #                       history['val_event_type_output_loss'])+1))

    # print_and_log(logger, "Highest event type validation loss was at "
    #               "epoch {}".format(np.argmin(
    #                       history['val_event_type_output_loss'])+1))

    # print_and_log(logger, "Lowest time delay validation loss was at "
    #               "epoch {}".format(np.argmin(
    #                       history['val_time_delay_output_loss'])+1))

    # print_and_log(logger, "Highest time delay validation loss was at "
    #               "epoch {}".format(np.argmax(
    #                       history['val_time_delay_output_loss'])+1))

    # print_and_log(logger, "Lowest user cluster validation loss was at "
    #               "epoch {}".format(np.argmin(
    #                       history['val_user_cluster_output_loss'])+1))

    # print_and_log(logger, "Highest user cluster validation loss was at "
    #               "epoch {}".format(np.argmax(
    #                       history['val_user_cluster_output_loss'])+1))

    print_and_log(logger, "======== Train ACCURACY")

    print_and_log(logger,
                  "Highest categorical_accuracy was at "
                  "epoch {}".format(np.argmax(
                          history['categorical_accuracy']
                          )+1))
    print_and_log(logger,
                  "Lowest categorical_accuracy was at "
                  "epoch {}".format(np.argmin(
                          history['categorical_accuracy']
                          )+1))

    # print_and_log(logger,
    #               "Highest event_type_output_categorical_accuracy was at "
    #               "epoch {}".format(np.argmax(
    #                       history['event_type_output_categorical_accuracy']
    #                       )+1))
    # print_and_log(logger,
    #               "Lowest time_delay_output_mean_squared_error was at "
    #               "epoch {}".format(np.argmin(
    #                       history['time_delay_output_mean_squared_error']
    #                       )+1))
    # print_and_log(
    #         logger,
    #         "Highest user_cluster_output_categorical_accuracy was at "
    #         "epoch {}".format(np.argmax(
    #                 history['user_cluster_output_categorical_accuracy']
    #                 )+1))

    print_and_log(logger, "======== Validation ACCURACY")

    print_and_log(logger,
                  "Highest val_categorical_accuracy was at "
                  "epoch {}".format(np.argmax(
                          history['val_categorical_accuracy']
                          )+1))

    print_and_log(logger,
                  "Lowest val_categorical_accuracy was at "
                  "epoch {}".format(np.argmin(
                          history['val_categorical_accuracy']
                          )+1))

    # print_and_log(logger,
    #               "Highest val_event_type_output_categorical_accuracy was at "
    #               "epoch {}".format(np.argmax(
    #                       history['val_event_type_output_categorical_accuracy']
    #                       )+1))
    # print_and_log(logger,
    #               "Lowest val_time_delay_output_mean_squared_error was at "
    #               "epoch {}".format(np.argmin(
    #                       history['val_time_delay_output_mean_squared_error']
    #                       )+1))
    # print_and_log(
    #         logger,
    #         "Highest val_user_cluster_output_categorical_accuracy was at "
    #         "epoch {}".format(np.argmax(
    #                 history['val_user_cluster_output_categorical_accuracy']
    #                 )))

    print_and_log(logger, "\nmodel history figures saved in {}".format(
            save_dir))
