#coding=utf-8
#coding=gbk

import pandas as pd
import numpy as np
import random
from util import *
import warnings
import os
import sys
from tqdm import tqdm
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.utilities.dataframe_functions import roll_time_series
cwd = os.path.dirname(__file__)
sys.path.append(os.path.dirname(cwd))
from global_setting.util import *

warnings.filterwarnings("ignore")


def filter_unaligned_moments(df):
    mask = (df[time] % 60 == 0)
    temp_df = df[mask]
    if temp_df.shape[0] == 0:
        first_row = df.iloc[0].copy()
        last_row = df.iloc[-1].copy()
        first_row[time] = 0
        last_row[time] = 60
        for feature in features:
            ori_value = last_row[feature]
            range_percent = 0.05
            range_value = ori_value * range_percent
            random.seed(0)
            last_row[feature] = ori_value + random.uniform(-range_value, range_value)
        temp_df.loc[0] = first_row
        temp_df.loc[1] = last_row
    if temp_df.shape[0] == 1:
        ori_time = temp_df.iloc[0][time]
        row = df.iloc[-1].copy() if ori_time == 0 else df.iloc[0].copy()
        row[time] = 60 if ori_time == 0 else ori_time - 60
        for feature in features:
            ori_value = row[feature]
            range_percent = 0.05
            range_value = ori_value * range_percent
            random.seed(0)
            row[feature] = ori_value + random.uniform(-range_value, range_value)
        temp_df = pd.concat([temp_df, pd.DataFrame([row])], ignore_index=True) if ori_time == 0 else \
            pd.concat([pd.DataFrame([row]), temp_df], ignore_index=True)
    return temp_df


#   数据读取
train_X, test_X = pd.read_csv(train_X_src_path, dtype={id: str}), \
    pd.read_csv(test_X_src_path, dtype={id: str})
print(len(train_X[id].unique()))
print(len(test_X[id].unique()))

#   筛除不对齐时刻
tqdm.pandas(desc='apply')
print("----------filter_unaligned_moments----------")
train_X = train_X.groupby(id, as_index=False).progress_apply(filter_unaligned_moments)
test_X = test_X.groupby(id, as_index=False).progress_apply(filter_unaligned_moments)
train_X[[time]+features] = train_X[[time]+features].astype(np.float64)
test_X[[time]+features] = test_X[[time]+features].astype(np.float64)
print(len(train_X[id].unique()))
print(len(test_X[id].unique()))

#   特征扩展
expand_feature_types = {
    'median': None, 'mean': None, 'standard_deviation': None,
    'variance': None, 'variance_larger_than_standard_deviation': None, 'variation_coefficient': None,
    'variation_coefficient': None, 'maximum': None, 'minimum': None,
    'mean_abs_change': None, 'mean_change': None, 'root_mean_square': None,
    'skewness': None, 'sample_entropy': None, 'mean_second_derivative_central': None
}
kind_to_fc_parameters = {feature: expand_feature_types for feature in features}

train_group_lengths = train_X.groupby('id').size().reset_index(name='Length')
test_group_lengths = test_X.groupby('id').size().reset_index(name='Length')
print(train_group_lengths['Length'].value_counts())
print(test_group_lengths['Length'].value_counts())

train_X_rolled = roll_time_series(train_X, column_id=id, column_sort=time,
                                  min_timeshift=1, max_timeshift=window_size-1)
test_X_rolled = roll_time_series(test_X, column_id=id, column_sort=time,
                                 min_timeshift=1, max_timeshift=window_size-1)
print(train_X_rolled.info())
print(test_X_rolled.info())
test_X = extract_features(test_X_rolled, column_id=id, column_sort=time,
                          kind_to_fc_parameters=kind_to_fc_parameters, impute_function=impute)
train_X = extract_features(train_X_rolled, column_id=id, column_sort=time,
                           kind_to_fc_parameters=kind_to_fc_parameters, impute_function=impute)

train_X.rename_axis([id, time], inplace=True)
train_X.reset_index(inplace=True)
test_X.rename_axis([id, time], inplace=True)
test_X.reset_index(inplace=True)

print(train_X.columns)
print(test_X.columns)
print(len(train_X.columns.values.tolist()))
print(len(test_X.columns.values.tolist()))

# train_X = extract_features(train_X, column_id=id, column_sort=time,
#                            default_fc_parameters=extraction_settings, impute_function=impute)
# test_X = extract_features(test_X, column_id=id, column_sort=time,
#                            default_fc_parameters=extraction_settings, impute_function=impute)

#   特征筛选
# print(len(train_X.columns.values.tolist()))
# print(len(test_X.columns.values.tolist()))
# train_y.set_index(id, inplace=True)
# train_X = select_features(train_X, train_y[label])
# test_X = test_X[train_X.columns.values.tolist()]
# print(train_X.columns)
# print(test_X.columns)
# print(len(train_X.columns.values.tolist()))
# print(len(test_X.columns.values.tolist()))

#   数据输出
train_X.to_csv(train_X_dst_path, index=None)
test_X.to_csv(test_X_dst_path, index=None)
