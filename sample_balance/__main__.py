#coding=utf-8
#coding=gbk

import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from util import *
import warnings
import os
import sys
cwd = os.path.dirname(__file__)
sys.path.append(os.path.dirname(cwd))
from global_setting.util import *

#   数据读取
train_X, train_y = pd.read_csv(train_X_src_path), pd.read_csv(train_set_src_path)[label]

#   SMOTE过采样
print('train set\'s labels distribution before resampling: %s' % Counter(train_y))
sm = SMOTE(random_state=0)
train_X, train_y = sm.fit_resample(train_X, train_y)
print('train set\'s labels distribution after resampling: %s' % Counter(train_y))

#   数据输出
train_X.to_csv(train_X_dst_path, index=None)
train_y.to_csv(train_y_dst_path, index=None)
