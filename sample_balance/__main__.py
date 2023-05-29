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
train_X_cp, train_y = pd.read_csv(train_X_cp_src_path), pd.read_csv(train_y_src_path)[label]
train_X_cp.drop(id, axis=1, inplace=True)

#   SMOTE过采样
print('train set\'s labels distribution before resampling: %s' % Counter(train_y))
sm = SMOTE(random_state=0)
train_X_sb, train_y_sb = sm.fit_resample(train_X_cp, train_y)
print('train set\'s labels distribution after resampling: %s' % Counter(train_y_sb))

#   数据输出
train_X_sb.to_csv(train_X_sb_dst_path, index=None)
train_y_sb.to_csv(train_y_sb_dst_path, index=None)
