#coding=utf-8
#coding=gbk

import numpy as np
import pandas as pd
from util import *
import warnings
import os
import sys
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import StandardScaler
cwd = os.path.dirname(__file__)
sys.path.append(os.path.dirname(cwd))
from global_setting.util import *

warnings.filterwarnings("ignore")

feature_medians = {}


def insert_missing_moments(df):
    time_min = df[time].min()
    df.set_index(time, inplace=True)
    id_value = df[id].mode().values[0]
    resampled_df = df.resample('H').asfreq()
    resampled_df[id].fillna(id_value, inplace=True)
    df = pd.concat([df, resampled_df])
    df.sort_index(inplace=True)
    df = df.reset_index().drop_duplicates(subset=[time], keep='last')
    df = df[df.apply(lambda row: row.time >= time_min, axis=1).cumsum().ge(1)]
    df[time] = (df[time] - time_min.floor('H')).dt.total_seconds() // 60
    return df


def fill_missing_values(df):
    for feature in features:
        X, Y = df[time], df[feature]
        value_mask = Y.notna()
        missing_mask = ~value_mask
        X_value, Y_value = X[value_mask].to_numpy(), Y[value_mask].to_numpy()
        valid_moments_num = len(X_value)
        invalid_moments_num = len(X) - valid_moments_num
        if valid_moments_num >= 2:
            X_missing = X[missing_mask].to_numpy()
            interp_func = UnivariateSpline(X_value, Y_value, k=min(3, valid_moments_num-1), ext=3)
            Y_missing = interp_func(X_missing)
        elif valid_moments_num == 1:
            Y_missing = np.full(shape=invalid_moments_num, fill_value=Y_value[0], dtype=np.float64)
        elif valid_moments_num == 0:
            Y_missing = np.full(shape=invalid_moments_num, fill_value=feature_medians[feature], dtype=np.float64)
        df[feature][missing_mask] = Y_missing
    return df


#   数据读取 & 标签构建
train_set, test_set = pd.read_csv(ori_train_set_src_path, dtype={id: str}), \
    pd.read_csv(ori_test_set_src_path, dtype={id: str})
train_y, test_y = train_set[[id, label]], test_set[[id, label]]
train_y.drop_duplicates(subset=[id], keep='last', inplace=True)
test_y.drop_duplicates(subset=[id], keep='last', inplace=True)
train_set.drop(columns=label, axis=1, inplace=True)
test_set.drop(columns=label, axis=1, inplace=True)
train_X, test_X = train_set, test_set

#   数据清洗
feature_medians = [train_X[feature].median() for feature in features]

train_X[time] = pd.to_datetime(train_X[time])
test_X[time] = pd.to_datetime(test_X[time])
tqdm.pandas(desc='apply')
print("----------insert missing moments----------")
train_X = train_X.groupby(id, as_index=False).progress_apply(insert_missing_moments)
test_X = test_X.groupby(id, as_index=False).progress_apply(insert_missing_moments)
print("----------filter valueless samples----------")
train_X = train_X.groupby(id, as_index=False).filter(lambda g: g[features].count().min() >= 2)
train_y = train_y[train_y[id].isin(train_X[id])]
print("----------fill missing values----------")
train_X = train_X.groupby(id, as_index=False).progress_apply(fill_missing_values)
test_X = test_X.groupby(id, as_index=False).progress_apply(fill_missing_values)

#   数据输出
train_X.to_csv(train_X_dst_path, index=None)
test_X.to_csv(test_X_dst_path, index=None)
train_y.to_csv(train_y_dst_path, index=None)
test_y.to_csv(test_y_dst_path, index=None)
