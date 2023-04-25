#coding=utf-8
#coding=gbk

import pandas as pd
from util import *
import warnings
import os
import sys
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
cwd = os.path.dirname(__file__)
sys.path.append(os.path.dirname(cwd))
from global_setting.util import *

warnings.filterwarnings("ignore")


def ICD9_to_OH_cols(row):
    values = [int(value) for value in eval(row['ICD9'])]
    value_to_count = {}
    for value in values:
        if value in value_to_count.keys():
            value_to_count[value] += 1
        else:
            value_to_count[value] = 1
    for value, count in value_to_count.items():
        col = 'ICD9_' + str(value)
        row[col] = count
    return row


#   数据读取
train_set, test_set = pd.read_csv(ori_train_set_src_path), pd.read_csv(ori_test_set_src_path)

#   数据清洗
train_set.dropna(subset=num_features, how='all', inplace=True)
train_set.set_index(pd.Index(range(train_set.shape[0])), inplace=True)

for feature in binary_cate_features:
    train_set[feature].fillna(0.0, inplace=True)
    test_set[feature].fillna(0.0, inplace=True)

for feature in multiple_cate_features:
    train_set[feature].fillna("NULL", inplace=True)
    test_set[feature].fillna("NULL", inplace=True)

for feature in list_features:
    train_set[feature].fillna("[]", inplace=True)
    test_set[feature].fillna("[]", inplace=True)

for feature in num_features:
    median_val = train_set[feature].median()
    train_set[feature].fillna(median_val, inplace=True)
    test_set[feature].fillna(median_val, inplace=True)

#   初始特征处理
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train_set = pd.DataFrame(OH_encoder.fit_transform(train_set[multiple_cate_features]))
OH_cols_train_set.columns = OH_encoder.get_feature_names(multiple_cate_features)
OH_cols_test_set = pd.DataFrame(OH_encoder.transform(test_set[multiple_cate_features]))
OH_cols_test_set.columns = OH_encoder.get_feature_names(multiple_cate_features)
train_set.drop(multiple_cate_features, axis=1, inplace=True)
test_set.drop(multiple_cate_features, axis=1, inplace=True)
train_set = pd.concat([train_set, OH_cols_train_set], axis=1)
test_set = pd.concat([test_set, OH_cols_test_set], axis=1)

ICD9_OH_cols = ['ICD9_' + str(i) for i in range(1, 1000)]
train_set = train_set.reindex(columns=train_set.columns.union(ICD9_OH_cols), fill_value=0)
test_set = test_set.reindex(columns=test_set.columns.union(ICD9_OH_cols), fill_value=0)
tqdm.pandas(desc='apply')
train_set = train_set.progress_apply(ICD9_to_OH_cols, axis=1)
test_set = test_set.progress_apply(ICD9_to_OH_cols, axis=1)
train_set.drop('ICD9', axis=1, inplace=True)
test_set.drop('ICD9', axis=1, inplace=True)

drop_cols = [col for col in set(train_set.columns).difference({label}) if train_set[col].nunique() == 1]
train_set.drop(columns=drop_cols, axis=1, inplace=True)
test_set.drop(columns=drop_cols, axis=1, inplace=True)

#   数据标准化
scaler = StandardScaler()
features_to_normalize = list(set(num_features + ICD9_OH_cols).intersection(train_set.columns))
train_set[features_to_normalize] = scaler.fit_transform(train_set[features_to_normalize])
test_set[features_to_normalize] = scaler.transform(test_set[features_to_normalize])

#   数据输出
train_set.to_csv(train_set_dst_path, index=None)
test_set.to_csv(test_set_dst_path, index=None)
