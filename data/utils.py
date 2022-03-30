import platform

import numpy as np
import pandas as pd
from datetime import datetime
import os
import six.moves.cPickle as pickle
import numpy as np
import h5py
import time
import pickle
import sys
import time
# import torch
# from einops import rearrange
from tqdm import tqdm
import random
from help_funcs import print_run_time


def reproducibility(seed=666):
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    print("random seed is fixed to: ",seed)

def running_window():
    return "windows" in running_platform()



def running_platform():
    return platform.system().lower()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def pretrain_shuffle(xc, xt, x_ext, y):
    """

    :param xc: (batch size, nb_flow, c, h, w)
    :param xt: (batch size, nb_flow, c, h, w)
    :param x_ext:
    :param y: (batch size, nb_flow, h, w)

    """
    xc, xt = list(map(lambda x: rearrange(x, "b n l h w -> l b n h w"), [xc, xt]))
    y = rearrange(y, "b n h w -> 1 b n h w")
    data = torch.cat([xc, xt, y], dim=0)  # l' b n h w
    his_len = len(data) - 1
    idx = torch.randint(0, his_len - 1, (1,))
    temp_y = data[-1].clone()  # normalize data[idx]
    data[-1] = data[idx]
    data[idx] = temp_y
    chunk_len = [len(xc), len(xt), 1]
    xc, xt, y = list(map(lambda x: rearrange(x, "l b n h w ->  b n l h w"), list(torch.split(data, chunk_len))))
    y = rearrange(y, "b n l h w -> b n h w")
    # renormalize y
    return xc, xt, x_ext, y



class MinMaxNormalization(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X


# def mean_squared_error(y_true, y_pred):
#     return K.mean(K.square(y_pred - y_true))
#
# def rmse(y_true, y_pred):
#     return mean_squared_error(y_true, y_pred) ** 0.5
#
# def mae(y_true, y_pred):
#     return K.mean(K.abs(y_pred - y_true))
#
#
# def compute(y_true, y_pred):
#     y_mse = np.mean(np.square(y_true-y_pred))
#     y_rmse = y_mse** 0.5
#     y_mae = np.mean(np.abs(y_true-y_pred))
#     idx = (y_true > 1e-6).nonzero()
#     y_mape = np.mean(np.abs((y_true[idx]-y_pred[idx])/y_true[idx]))
#     return y_rmse, y_mae, y_mape

def compute(y_true, y_pred):
    """
    support computing Error metrics on two data type, torch.Tensor and np.ndarray.
    """
    if isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        backend = torch
    elif isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        backend = np
    y_mse = backend.mean((y_true - y_pred) ** 2)
    y_rmse = y_mse ** 0.5
    y_mae = backend.mean(backend.abs(y_true - y_pred))
    idx = (y_true > 10)
    y_mape = backend.mean(backend.abs((y_true[idx] - y_pred[idx]) / y_true[idx]))
    ape = backend.sum(backend.abs((y_true[idx] - y_pred[idx]) / y_true[idx]))
    reshaped_y_true = y_true.reshape(-1)
    cell_mean = backend.mean(reshaped_y_true, 0)
    relative_error = y_mae / cell_mean

    y_rmse, y_mae, y_mape, ape = y_rmse.item(), y_mae.item(), y_mape.item(), ape.item()
    return y_rmse, y_mae, y_mape, ape


def remove_incomplete_days(data, timestamps, T=48):
    print("T",T)
    # remove a certain day which has not 48 timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i + T - 1 < len(timestamps) and int(timestamps[i + T - 1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps


def load_stdata(fname):
    # print('fname:', fname)
    f = h5py.File(fname, 'r')
    data = f['data'][:]
    timestamps = f['date'][:]
    f.close()
    return data, timestamps


def string2timestamp(strings, T=48):
    '''
    strings: list, eg. ['2017080912','2017080913']
    return: list, eg. [Timestamp('2017-08-09 05:30:00'), Timestamp('2017-08-09 06:00:00')]
    '''
    timestamps = []
    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:]) 
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot),
                                                minute=(slot % num_per_T) * int(60.0 * time_per_slot))))

    return timestamps


class STMatrix(object):
    """docstring for STMatrix"""

    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.data_1 = data[:, 0, :, :]
        self.data_2 = data[:, 1, :, :]
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i - 1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]

    def get_matrix_1(self, timestamp):  # in_flow
        ori_matrix = self.data_1[self.get_index[timestamp]]
        new_matrix = ori_matrix[np.newaxis, :]
        # print("new_matrix shape:",new_matrix.shape) #(1, 32, 32)
        return new_matrix

    def get_matrix_2(self, timestamp):  # out_flow
        ori_matrix = self.data_2[self.get_index[timestamp]]
        new_matrix = ori_matrix[np.newaxis, :]
        # print("new_matrix shape:",new_matrix.shape) #(1, 32, 32)
        return new_matrix

    def save(self, fname):
        pass

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset_3D(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1,
                          prediction_offset=0):
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness + 1),
                   [PeriodInterval * self.T * j for j in range(1, len_period + 1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend + 1)]]

        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue

            # closeness
            c_1_depends = list(depends[0])  # in_flow
            c_1_depends.sort(reverse=True)
            # print('----- c_1_depends:',c_1_depends)

            c_2_depends = list(depends[0])  # out_flow
            c_2_depends.sort(reverse=True)
            # print('----- c_2_depends:',c_2_depends)

            x_c_1 = [self.get_matrix_1(self.pd_timestamps[i] - j * offset_frame) for j in
                     c_1_depends]  # [(1,32,32),(1,32,32),(1,32,32)] in
            x_c_2 = [self.get_matrix_2(self.pd_timestamps[i] - j * offset_frame) for j in
                     c_2_depends]  # [(1,32,32),(1,32,32),(1,32,32)] out

            x_c_1_all = np.vstack(x_c_1)  # x_c_1_all.shape  (3, 32, 32)
            x_c_2_all = np.vstack(x_c_2)  # x_c_1_all.shape  (3, 32, 32)

            x_c_1_new = x_c_1_all[np.newaxis, :]  # (1, 3, 32, 32)
            x_c_2_new = x_c_2_all[np.newaxis, :]  # (1, 3, 32, 32)

            x_c = np.vstack([x_c_1_new, x_c_2_new])  # (2, 3, 32, 32)

            # period
            p_depends = list(depends[1])
            if (len(p_depends) > 0):
                p_depends.sort(reverse=True)
                # print('----- p_depends:',p_depends)

                x_p_1 = [self.get_matrix_1(self.pd_timestamps[i] - j * offset_frame) for j in p_depends]
                x_p_2 = [self.get_matrix_2(self.pd_timestamps[i] - j * offset_frame) for j in p_depends]

                x_p_1_all = np.vstack(x_p_1)  # [(3,32,32),(3,32,32),...]
                x_p_2_all = np.vstack(x_p_2)  # [(3,32,32),(3,32,32),...]

                x_p_1_new = x_p_1_all[np.newaxis, :]  # (1, 3, 32, 32)
                x_p_2_new = x_p_2_all[np.newaxis, :]  # (1, 3, 32, 32)

                x_p = np.vstack([x_p_1_new, x_p_2_new])  # (2, 3, 32, 32)

            # trend
            t_depends = list(depends[2])
            if (len(t_depends) > 0):
                t_depends.sort(reverse=True)

                x_t_1 = [self.get_matrix_1(self.pd_timestamps[i] - j * offset_frame) for j in t_depends]
                x_t_2 = [self.get_matrix_2(self.pd_timestamps[i] - j * offset_frame) for j in t_depends]

                x_t_1_all = np.vstack(x_t_1)  # [(3,32,32),(3,32,32),...]
                x_t_2_all = np.vstack(x_t_2)  # [(3,32,32),(3,32,32),...]

                x_t_1_new = x_t_1_all[np.newaxis, :]  # (1, 3, 32, 32)
                x_t_2_new = x_t_2_all[np.newaxis, :]  # (1, 3, 32, 32)

                x_t = np.vstack([x_t_1_new, x_t_2_new])  # (2, 3, 32, 32)

            y = self.get_matrix(self.pd_timestamps[i + prediction_offset])

            if len_closeness > 0:
                XC.append(x_c)
            if len_period > 0:
                XP.append(x_p)
            if len_trend > 0:
                XT.append(x_t)
            Y.append(y)
            timestamps_Y.append(self.timestamps[i + prediction_offset])
            i += 1

        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        print("3D matrix - XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y, timestamps_Y

def timestamp2array(timestamps, T):
    """
    把时间戳的序列中的每一个时间戳转成特征数组，考虑了星期和小时，
    时间戳: numpy.datetime64('2013-07-01T00:00:00.000000000')

    Args:
        timestamps: 时间戳序列
        t: 一天有多少个时间步

    Returns:
        np.ndarray: 特征数组，shape: (len(timestamps), ext_dim)
    """
    # print("timestamps: ", timestamps)
    temp = []
    for t in timestamps:
        t = t.astype(np.str)
        temp.append(np.datetime64(datetime(int(t[:4]),int(t[4:6]),int((t[6:8])),int((t[8:])))))
    timestamps = np.array(temp)
    vec_wday = [time.strptime(
        str(t)[:10], '%Y-%m-%d').tm_wday for t in timestamps]
    vec_hour = [time.strptime(str(t)[11:13], '%H').tm_hour for t in timestamps]
    vec_minu = [time.strptime(str(t)[14:16], '%M').tm_min for t in timestamps]
    ret = []
    for idx, wday in enumerate(vec_wday):
        # day
        v = [0 for _ in range(7)]
        v[wday] = 1
        if wday >= 5:  # 0是周一, 6是周日
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday len(v)=8
        # hour
        v += [0 for _ in range(T)]  # len(v)=8+T
        hour = vec_hour[idx]
        minu = vec_minu[idx]
        # 24*60/T 表示一个时间步是多少分钟
        # hour * 60 + minu 是从0:0开始到现在是多少分钟，相除计算是第几个时间步
        # print(hour, minu, T, (hour * 60 + minu) / (24 * 60 / T))
        v[int((hour * 60 + minu) / (24 * 60 / T))] = 1
        # +8是因为v前边有表示星期的8位
        if hour >= 18 or hour < 6:
            v.append(0)  # night
        else:
            v.append(1)  # day
        ret.append(v)  # len(v)=7+1+T+1=T+9
    return np.asarray(ret)



def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0
    vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]  # python3
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)