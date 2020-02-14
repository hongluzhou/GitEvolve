#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:59:21 2019

@author: zhouhonglu
"""
import os
import pdb
import argparse
import time

from models.multitask_repeat_no_action import create_config
from models.multitask_repeat_no_action import test
from models.multitask_repeat_no_action import evaluate


if __name__ == '__main__':
    time_start = time.time()

    # get config
    config = create_config.create_config()

    # create dirs if not exist
    if not os.path.exists(config['exp_save_dir']):
        os.makedirs(config['exp_save_dir'])

    # argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('run_option', type=str, default='test',
                        help='test or eval')
    parser.add_argument('--create_dataset', type=int, default=1,
                        help='1 to create, 0 not to create')

    args = parser.parse_args()

    if args.run_option == "test":

        process_test = True
        process_eval = False
        if args.create_dataset == 1:
            process_create_dataset = True
        else:
            process_create_dataset = False

    elif args.run_option == "eval":

        process_test = False
        process_eval = True

    else:
        print("please tell me to train or test or eval!")
        pdb.set_trace()

    for key in config:
        print("{}: {}".format(key, config[key]))
    print("config loaded!\n")

    # process
    if process_test:
        test.test(config, process_create_dataset)
        evaluate.eval(config)

    if process_eval:
        evaluate.eval(config)

    print('\n\nFinished! ' +
          "======================================================")
    print("\nThe entire process took {} s".format(
            round(time.time() - time_start), 2))
    print("\nThe experiment folder: {}".format(config['exp_save_dir']))
