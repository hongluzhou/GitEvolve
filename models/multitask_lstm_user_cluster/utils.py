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


def weighted_et_bce(y_true, y_pred):
    # replace pos and neg with lists corresponding to the specific task.
    # copy paste this loss function for both event type and user cluster.
    # manually define the lists like below. Don't try to get it as input.
    """
    each_event_type_total_count
    {1: 6503, 4: 30762, 12: 16077, 5: 50513, 3: 18727, 8: 8421, 9: 39632,
    2: 3230, 6: 6593, 10: 1923, 7: 5506}
    """
    pos = np.array([0.034611229089825266, 0.017191184062761128,
                    0.09967161112796522, 0.16372606939277332,
                    0.26884776487995443, 0.03509024041045948,
                    0.029304848126799617, 0.04481949256734101,
                    0.21093529621527835, 0.010234875217550975,
                    0, 0.08556738890929122])
    neg = np.array([0.9653887709101747, 0.9828088159372389,
                    0.9003283888720348, 0.8362739306072267,
                    0.7311522351200456, 0.9649097595895405,
                    0.9706951518732004, 0.955180507432659,
                    0.7890647037847217, 0.989765124782449,
                    1.0, 0.9144326110907088])
    pos_k = K.variable(pos)
    neg_k = K.variable(neg)
    ones_k = K.ones_like(y_true)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    interloss = -K.mean(neg_k * y_true * K.log(y_pred) + pos_k * (
            ones_k - y_true) * K.log(ones_k - y_pred), axis=0)

    finalloss = K.sum(interloss)
    return finalloss


def weighted_uc_bce(y_true, y_pred):
    # replace pos and neg with lists corresponding to the specific task.
    # copy paste this loss function for both event type and user cluster.
    # manually define the lists like below. Don't try to get it as input.
    """
    each_cluster_total_event_count
    {90: 1058, 100: 16077, 3: 3571, 8: 2592, 96: 3769, 92: 140, 20: 1432,
    45: 1179, 95: 7337, 58: 963, 1: 9400, 47: 53, 2: 1890, 88: 7125,
    64: 2804, 22: 5979, 33: 10207, 10: 566, 55: 520, 21: 317, 36: 2726,
    70: 1844, 39: 86, 73: 866, 11: 2353, 44: 808, 18: 1638, 91: 2732,
    48: 5387, 26: 546, 42: 816, 80: 406, 94: 671, 34: 3121, 62: 710,
    99: 280, 81: 466, 68: 2582, 74: 1196, 69: 1839, 37: 2440, 79: 1453,
    4: 1545, 67: 939, 77: 801, 19: 827, 46: 73, 9: 537, 13: 1037, 12: 572,
    0: 1032, 56: 860, 78: 114, 60: 1126, 61: 617, 85: 1243, 72: 1398,
    49: 435, 28: 1553, 5: 530, 27: 581, 52: 374, 59: 536, 54: 1331, 29: 948,
    93: 1002, 6: 222, 32: 433, 31: 52, 75: 738, 82: 54, 86: 81, 17: 2475,
    98: 38, 97: 1771, 83: 1514, 76: 3768, 66: 5254, 40: 5941, 57: 2505,
    71: 2340, 43: 6461, 23: 3365, 53: 235, 51: 231, 7: 66, 16: 22, 35: 8,
    30: 2590, 14: 2893, 84: 223, 89: 13, 15: 403, 87: 15, 50: 1, 63: 4,
    41: 11, 25: 5562, 24: 33, 38: 5341, 65: 1268}
    """
    pos = np.array([0.005492663143272286, 0.05003007126623982, 0.01005923773331843, 0.019006104733164084, 0.00822302767088729, 0.002820844443734798, 0.0011815612575643872, 0.00035127496846508807, 0.013795526034265277, 0.0028581008797841255, 0.0030124489719884824, 0.012523484860581094, 0.0030443830600307632, 0.00551927488330752, 0.01539755278438636, 0.0021449062468398557, 0.00011709165615502935, 0.013172811317440802, 0.00871800603554264, 0.004401581801827694, 0.007621602346091001, 0.001687184318233832, 0.03182231873413275, 0.017909701043712443, 0.00017563748423254404, 0.029602899615194238, 0.002906002011847547, 0.003092284192094184, 0.008265606454943663, 0.005045585910680356, 0.013784881338251183, 0.000276762096366433, 0.0023045766870512597, 0.05432520610792657, 0.016611048129993027, 4.257878405637431e-05, 0.014508720667209546, 0.012986529137194164, 0.0284266607056369, 0.00045772192860602385, 0.03162006950986497, 5.8545828077514676e-05, 0.00434303597375018, 0.034387690473529305, 0.0043004571896938055, 0.006275048300308164, 0.0003885314045144156, 0.0002820844443734798, 0.028671488713961052, 0.002315221383065353, 5.3223480070467884e-06, 0.0012294623896278083, 0.001990558154635499, 0.0012507517816559954, 0.007084045197379276, 0.0027676209636643303, 0.004577219286060238, 0.013332481757652205, 0.005125421130786058, 0.002852778531777079, 0.005992963855934684, 0.0032838887203478686, 0.00377886708500322, 2.1289392028187154e-05, 0.014923863811759195, 0.006748737272935328, 0.027963616429023827, 0.0049976847786169344, 0.013742302554194808, 0.009787797984959044, 0.009814409724994278, 0.012454294336489485, 0.007440642513851411, 0.004609153374102519, 0.006365528216427959, 0.00392789282920053, 0.0200546072905523, 0.0042632007536444775, 0.0006067476728033339, 0.007733371654238984, 0.002160873290860996, 0.0024802141712838037, 0.00028740679238052657, 0.008058034882668838, 0.0011868836055714338, 0.0066156785727591585, 0.0004311101885707899, 7.983522010570184e-05, 0.03792172955020837, 6.919052409160825e-05, 0.0056310441914555025, 0.014540654755251826, 0.0007451287209865505, 0.005332992703060882, 0.0035712955127283953, 0.03905006732770229, 0.02005992963855935, 0.009425878320479864, 0.00020224922426777798, 0.001490257441973101, 0.08556738890929122])
    neg = np.array([0.9945073368567278, 0.9499699287337602, 0.9899407622666816, 0.9809938952668359, 0.9917769723291127, 0.9971791555562652, 0.9988184387424356, 0.9996487250315349, 0.9862044739657347, 0.9971418991202159, 0.9969875510280115, 0.9874765151394189, 0.9969556169399693, 0.9944807251166925, 0.9846024472156136, 0.9978550937531602, 0.999882908343845, 0.9868271886825593, 0.9912819939644574, 0.9955984181981723, 0.992378397653909, 0.9983128156817662, 0.9681776812658672, 0.9820902989562875, 0.9998243625157675, 0.9703971003848058, 0.9970939979881525, 0.9969077158079058, 0.9917343935450563, 0.9949544140893196, 0.9862151186617488, 0.9997232379036336, 0.9976954233129487, 0.9456747938920734, 0.983388951870007, 0.9999574212159437, 0.9854912793327905, 0.9870134708628059, 0.9715733392943631, 0.999542278071394, 0.968379930490135, 0.9999414541719225, 0.9956569640262498, 0.9656123095264707, 0.9956995428103061, 0.9937249516996919, 0.9996114685954856, 0.9997179155556265, 0.9713285112860389, 0.9976847786169346, 0.9999946776519929, 0.9987705376103722, 0.9980094418453646, 0.998749248218344, 0.9929159548026207, 0.9972323790363357, 0.9954227807139397, 0.9866675182423478, 0.9948745788692139, 0.9971472214682229, 0.9940070361440653, 0.9967161112796521, 0.9962211329149968, 0.9999787106079718, 0.9850761361882409, 0.9932512627270647, 0.9720363835709762, 0.995002315221383, 0.9862576974458052, 0.990212202015041, 0.9901855902750057, 0.9875457056635105, 0.9925593574861485, 0.9953908466258975, 0.993634471783572, 0.9960721071707994, 0.9799453927094477, 0.9957367992463555, 0.9993932523271967, 0.9922666283457611, 0.997839126709139, 0.9975197858287161, 0.9997125932076195, 0.9919419651173311, 0.9988131163944286, 0.9933843214272409, 0.9995688898114292, 0.9999201647798943, 0.9620782704497917, 0.9999308094759084, 0.9943689558085445, 0.9854593452447482, 0.9992548712790135, 0.9946670072969391, 0.9964287044872716, 0.9609499326722977, 0.9799400703614407, 0.9905741216795202, 0.9997977507757322, 0.9985097425580269, 0.9144326110907087])
    pos_k = K.variable(pos)
    neg_k = K.variable(neg)
    ones_k = K.ones_like(y_true)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    interloss = -K.mean(neg_k * y_true * K.log(y_pred) + pos_k * (
            ones_k - y_true) * K.log(ones_k - y_pred), axis=0)

    finalloss = K.sum(interloss)
    return finalloss


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
    elif time_delay_normalization_func == 'log10_xplus1/10':
        return (np.log10(x + 1)/10)


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
                                one_chain_user):
    one_chain_event_new = []
    one_chain_time_new = []
    one_chain_user_new = []
    for i in range(len(one_chain_time)):
        if one_chain_event[i] == config['empty_event_type'] or (
                one_chain_event[i] == config['eventtype_2_id']['<soc>']):
            one_chain_event_new.append(one_chain_event[i])
            one_chain_user_new.append(one_chain_user[i])
            one_chain_time_new.append(one_chain_time[i])
        elif one_chain_event[i-1] == config['eventtype_2_id']['<soc>']:
            one_chain_event_new.append(one_chain_event[i])
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
                one_chain_user_new.append("no_event_user")
                one_chain_time_new.append(
                        one_chain_time[i-1] + 30.0*24*3600)
                time_delay = get_time_delay(one_chain_time_new[-1],
                                            one_chain_time[i],
                                            'float')[1]
                while time_delay > 30*24:
                    one_chain_event_new.append(config['eventtype_2_id'][
                            "no_event_for_1month"])
                    one_chain_user_new.append("no_event_user")
                    one_chain_time_new.append(
                            one_chain_time_new[-1] + 30.0*24*3600)
                    time_delay = get_time_delay(one_chain_time_new[-1],
                                                one_chain_time[i],
                                                'float')[1]
            one_chain_event_new.append(one_chain_event[i])
            one_chain_user_new.append(one_chain_user[i])
            one_chain_time_new.append(one_chain_time[i])

    return (one_chain_event_new, one_chain_time_new, one_chain_user_new)


def insert_no_event_for_a_chain_new(config, one_chain_event,
                                    one_chain_time,
                                    one_chain_user, sim_start):
    one_chain_event_new = []
    one_chain_time_new = []
    one_chain_user_new = []
    if len(one_chain_time) == 0:
        print("Wrong! "
              "This implementation does not support empty chain time!")
        pdb.set_trace()

    for i in range(len(one_chain_time)):
        if one_chain_event[i] == config['empty_event_type'] or (
                one_chain_event[i] == config['eventtype_2_id']['<soc>']):
            one_chain_event_new.append(one_chain_event[i])
            one_chain_user_new.append(one_chain_user[i])
            one_chain_time_new.append(one_chain_time[i])
        elif one_chain_event[i-1] == config['eventtype_2_id']['<soc>']:
            one_chain_event_new.append(one_chain_event[i])
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
                one_chain_user_new.append("no_event_user")
                one_chain_time_new.append(
                        one_chain_time[i-1] + 30.0*24*3600)
                time_delay = get_time_delay(one_chain_time_new[-1],
                                            one_chain_time[i],
                                            'float')[1]
                while time_delay > 30*24:
                    one_chain_event_new.append(config['eventtype_2_id'][
                            "no_event_for_1month"])
                    one_chain_user_new.append("no_event_user")
                    one_chain_time_new.append(
                            one_chain_time_new[-1] + 30.0*24*3600)
                    time_delay = get_time_delay(one_chain_time_new[-1],
                                                one_chain_time[i],
                                                'float')[1]
            one_chain_event_new.append(one_chain_event[i])
            one_chain_user_new.append(one_chain_user[i])
            one_chain_time_new.append(one_chain_time[i])

    # have inserted no_event before last true event before sim

    # insert more event to reach true end of training
    if (sim_start - one_chain_time_new[-1]) > (30.0*24*3600):
        num2insert = int((sim_start - one_chain_time_new[-1])/(30.0*24*3600))
        for each in range(num2insert):
            one_chain_event_new.append(config['eventtype_2_id'][
                    "no_event_for_1month"])
            one_chain_user_new.append("no_event_user")
            one_chain_time_new.append(
                    one_chain_time_new[-1] + 30.0*24*3600)

    return (one_chain_event_new, one_chain_time_new, one_chain_user_new)


def insert_no_event_for_a_sim_GTchain(config,
                                      one_chain_event, one_chain_time,
                                      one_chain_user,
                                      input_last_time,
                                      sim_end):
    one_chain_event_new = []
    one_chain_time_new = []
    one_chain_user_new = []
    time_delay = get_time_delay(input_last_time,
                                one_chain_time[0],
                                'float')[1]
    if time_delay > 30*24:  # time delay is greater than 30 days
        one_chain_event_new.append(config['eventtype_2_id'][
                "no_event_for_1month"])
        one_chain_user_new.append("no_event_user")
        one_chain_time_new.append(
                input_last_time + 30.0*24*3600)
        time_delay = get_time_delay(one_chain_time_new[-1],
                                    one_chain_time[0],
                                    'float')[1]
        while time_delay > 30*24:
            one_chain_event_new.append(config['eventtype_2_id'][
                "no_event_for_1month"])
            one_chain_user_new.append("no_event_user")
            one_chain_time_new.append(
                    one_chain_time_new[-1] + 30.0*24*3600)
            time_delay = get_time_delay(one_chain_time_new[-1],
                                        one_chain_time[0],
                                        'float')[1]
    one_chain_event_new.append(one_chain_event[0])
    one_chain_user_new.append(one_chain_user[0])
    one_chain_time_new.append(one_chain_time[0])

    for i in range(1, len(one_chain_time)):
        time_delay = get_time_delay(one_chain_time_new[-1],
                                    one_chain_time[i],
                                    'float')[1]
        if time_delay > 30*24:  # time delay is greater than 30 days
            one_chain_event_new.append(config['eventtype_2_id'][
                    "no_event_for_1month"])
            one_chain_user_new.append("no_event_user")
            one_chain_time_new.append(
                    one_chain_time[i-1] + 30.0*24*3600)
            time_delay = get_time_delay(one_chain_time_new[-1],
                                        one_chain_time[i],
                                        'float')[1]
            while time_delay > 30*24:
                one_chain_event_new.append(config['eventtype_2_id'][
                    "no_event_for_1month"])
                one_chain_user_new.append("no_event_user")
                one_chain_time_new.append(
                        one_chain_time_new[-1] + 30.0*24*3600)
                time_delay = get_time_delay(one_chain_time_new[-1],
                                            one_chain_time[i],
                                            'float')[1]
        one_chain_event_new.append(one_chain_event[i])
        one_chain_user_new.append(one_chain_user[i])
        one_chain_time_new.append(one_chain_time[i])

    # have inserted no_event between before last event in sim

    # insert more event to reach true end of sim
    if (sim_end - one_chain_time_new[-1]) > (30.0*24*3600):
        num2insert = int((sim_end - one_chain_time_new[-1])/(30.0*24*3600))
        for each in range(num2insert):
            one_chain_event_new.append(config['eventtype_2_id'][
                    "no_event_for_1month"])
            one_chain_user_new.append("no_event_user")
            one_chain_time_new.append(
                    one_chain_time_new[-1] + 30.0*24*3600)

    return (one_chain_event_new, one_chain_time_new, one_chain_user_new)


def insert_no_event_for_a_GTchain_who_has_no_event_at_all(
        config, one_chain_event_new, one_chain_time_new, one_chain_user_new,
        input_last_time, sim_end):

    # insert more event to reach true end of sim
    if (sim_end - input_last_time) > (30.0*24*3600):
        num2insert = int((sim_end - input_last_time)/(30.0*24*3600))
        for each in range(num2insert):
            one_chain_event_new.append(config['eventtype_2_id'][
                    "no_event_for_1month"])
            one_chain_user_new.append("no_event_user")
            try:
                one_chain_time_new.append(
                        one_chain_time_new[-1] + 30.0*24*3600)
            except IndexError:
                one_chain_time_new.append(
                        input_last_time + 30.0*24*3600)
        # pdb.set_trace()
    return (one_chain_event_new, one_chain_time_new, one_chain_user_new)


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
