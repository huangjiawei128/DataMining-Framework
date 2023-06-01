<<<<<<< HEAD
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
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from tsfresh import select_features
cwd = os.path.dirname(__file__)
sys.path.append(os.path.dirname(cwd))
from global_setting.util import *

warnings.filterwarnings("ignore")


def ewm_mean(col):
    alpha = 2 / (len(col) + 1)
    return col.ewm(alpha=alpha, adjust=False).mean().iloc[-1]


#   数据读取
train_X, test_X = pd.read_csv(train_X_src_path, dtype={id: str}), \
    pd.read_csv(test_X_src_path, dtype={id: str})
train_X.drop(time, axis=1, inplace=True)
test_X.drop(time, axis=1, inplace=True)
train_y = pd.read_csv(train_y_src_path, dtype={id: str})

#   数据标准化
cur_features = train_X.columns.values.tolist()
cur_features.remove(id)
scaler = StandardScaler()
train_X[cur_features] = scaler.fit_transform(train_X[cur_features])
test_X[cur_features] = scaler.transform(test_X[cur_features])

#   主成分分析降维
PCA_transfer = PCA(n_components=PCA_n_components, random_state=0)
train_X = pd.concat([train_X[id], pd.DataFrame(PCA_transfer.fit_transform(train_X[cur_features]))], axis=1)
test_X = pd.concat([test_X[id], pd.DataFrame(PCA_transfer.transform(test_X[cur_features]))], axis=1)

#   时间序列压缩
train_X_cp = train_X.groupby(id, as_index=False).agg(ewm_mean)
test_X_cp = test_X.groupby(id, as_index=False).agg(ewm_mean)

#   特征子集选择
cur_features = train_X.columns.values.tolist()
cur_features.remove(id)
train_X_cp.set_index(id, inplace=True)
train_y.set_index(id, inplace=True)
train_X_cp = select_features(train_X_cp, train_y[label].astype(bool))
selected_features = train_X_cp.columns.values.tolist()
features_to_remove = list(set(cur_features) - set(selected_features))
train_X_cp.sort_index(axis=1, inplace=True)
train_X_cp.reset_index(inplace=True)
test_X_cp.drop(features_to_remove, axis=1, inplace=True)
train_X.drop(features_to_remove, axis=1, inplace=True)
test_X.drop(features_to_remove, axis=1, inplace=True)

#   数据输出
train_X.to_csv(train_X_dst_path, index=None)
test_X.to_csv(test_X_dst_path, index=None)
train_X_cp.to_csv(train_X_cp_dst_path, index=None)
test_X_cp.to_csv(test_X_cp_dst_path, index=None)
=======
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
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from tsfresh import select_features
cwd = os.path.dirname(__file__)
sys.path.append(os.path.dirname(cwd))
from global_setting.util import *

warnings.filterwarnings("ignore")


def ewm_mean(col):
    alpha = 2 / (len(col) + 1)
    return col.ewm(alpha=alpha, adjust=False).mean().iloc[-1]


#   数据读取
train_X, test_X = pd.read_csv(train_X_src_path, dtype={id: str}), \
    pd.read_csv(test_X_src_path, dtype={id: str})
train_X.drop(time, axis=1, inplace=True)
test_X.drop(time, axis=1, inplace=True)
train_y = pd.read_csv(train_y_src_path, dtype={id: str})

#   数据标准化
cur_features = train_X.columns.values.tolist()
cur_features.remove(id)
scaler = StandardScaler()
train_X[cur_features] = scaler.fit_transform(train_X[cur_features])
test_X[cur_features] = scaler.transform(test_X[cur_features])

#   主成分分析降维
PCA_transfer = PCA(n_components=PCA_n_components, random_state=0)
train_X = pd.concat([train_X[id], pd.DataFrame(PCA_transfer.fit_transform(train_X[cur_features]))], axis=1)
test_X = pd.concat([test_X[id], pd.DataFrame(PCA_transfer.transform(test_X[cur_features]))], axis=1)

#   时间序列压缩
train_X_cp = train_X.groupby(id, as_index=False).agg(ewm_mean)
test_X_cp = test_X.groupby(id, as_index=False).agg(ewm_mean)

#   特征子集选择
cur_features = train_X.columns.values.tolist()
cur_features.remove(id)
train_X_cp.set_index(id, inplace=True)
train_y.set_index(id, inplace=True)
train_X_cp = select_features(train_X_cp, train_y[label].astype(bool))
selected_features = train_X_cp.columns.values.tolist()
features_to_remove = list(set(cur_features) - set(selected_features))
train_X_cp.sort_index(axis=1, inplace=True)
train_X_cp.reset_index(inplace=True)
test_X_cp.drop(features_to_remove, axis=1, inplace=True)
train_X.drop(features_to_remove, axis=1, inplace=True)
test_X.drop(features_to_remove, axis=1, inplace=True)

#   数据输出
train_X.to_csv(train_X_dst_path, index=None)
test_X.to_csv(test_X_dst_path, index=None)
train_X_cp.to_csv(train_X_cp_dst_path, index=None)
test_X_cp.to_csv(test_X_cp_dst_path, index=None)
>>>>>>> 2a6ebe51ab223003d660b142310d9fe2ef72b085
