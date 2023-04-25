#coding=utf-8
#coding=gbk

import pandas as pd
import numpy as np
from util import *
import warnings
import os
import sys
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
cwd = os.path.dirname(__file__)
sys.path.append(os.path.dirname(cwd))
from global_setting.util import *

warnings.filterwarnings("ignore")

#   数据读取
train_set, test_set = pd.read_csv(train_set_src_path), pd.read_csv(test_set_src_path)
features = list(set(train_set.columns).difference({label}))
train_X, test_X = train_set[features], test_set[features]
train_y = pd.read_csv(train_set_src_path)[label]

#   主成分分析降维
PCA_transfer = PCA(n_components=PCA_n_components, random_state=0)
train_X = pd.DataFrame(PCA_transfer.fit_transform(train_X))
test_X = pd.DataFrame(PCA_transfer.transform(test_X))

#   特征子集选择
corrs = {feature: abs(spearmanr(train_X[feature], train_y)[0]) for feature in tqdm(train_X.columns)}
features_to_remove = [feature for feature in corrs.keys() if corrs[feature] < spearman_threshold]
train_X.drop(features_to_remove, axis=1, inplace=True)
test_X.drop(features_to_remove, axis=1, inplace=True)

#   数据输出
train_X.to_csv(train_X_dst_path, index=None)
test_X.to_csv(test_X_dst_path, index=None)
