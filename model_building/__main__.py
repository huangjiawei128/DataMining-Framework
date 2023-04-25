#coding=utf-8
#coding=gbk

import pandas as pd
import numpy as np
import pickle
import argparse
from util import *
import warnings
import os
import sys
cwd = os.path.dirname(__file__)
sys.path.append(os.path.dirname(cwd))
from global_setting.util import *

warnings.filterwarnings("ignore")


def model_building_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regressor', help='the regressor to be trained', type=str)
    return parser


#   参数处理
args = model_building_parser().parse_args()
regressor_name = args.regressor

#   数据读取
train_X, train_y = pd.read_csv(train_X_src_path), pd.read_csv(train_set_src_path)[label]
train_X, train_y = train_X.to_numpy(), train_y.to_numpy()

#   回归器训练
regressor_params = regressor_params_map[regressor_name]
regressor = regressor_map[regressor_name](train_X, train_y, regressor_params)

#   回归器输出
regressor_file = regressor_name
regressor_params = sorted(regressor_params.items(), key=lambda x: x[0], reverse=False)
for param, value in regressor_params:
    regressor_file += '_%s_%s' % (param, str(value))
regressor_file = '%s/%s.h5' % (regressor_dst_dir, regressor_file)
print(regressor_file)
with open(regressor_file, 'wb') as rf:
    pickle.dump(regressor, rf, protocol=pickle.HIGHEST_PROTOCOL)
