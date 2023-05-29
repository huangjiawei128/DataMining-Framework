#coding=utf-8
#coding=gbk

import pandas as pd
import numpy as np
import pickle
import argparse
from collections import Counter
from imblearn.over_sampling import SMOTE
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
    parser.add_argument('--classifier', help='the classifier to be trained', type=str)
    return parser


#   参数处理
args = model_building_parser().parse_args()
classifier_name = args.classifier
classifier_info = classifier_map[classifier_name]

#   数据读取
need_cp_samples = classifier_info['need_cp_samples']
if need_cp_samples:
    train_X, train_y = pd.read_csv(train_X_cp_src_path), pd.read_csv(train_y_src_path)[label]
    train_X.drop(id, axis=1, inplace=True)
    #   SMOTE过采样
    print('train set\'s labels distribution before resampling: %s' % Counter(train_y))
    sm = SMOTE(random_state=0)
    train_X, train_y = sm.fit_resample(train_X, train_y)
    print('train set\'s labels distribution after resampling: %s' % Counter(train_y))
    train_X, train_y = train_X.to_numpy(), train_y.to_numpy()
else:
    train_X, train_y = pd.read_csv(train_X_src_path), pd.read_csv(train_y_src_path)
    #   TODO

classifier_func, classifier_params_lst = classifier_info['func'], classifier_info['params_lst']
for classifier_params in classifier_params_lst:
    # 分类器训练
    print("---------- classifier training starts ----------")
    print(classifier_params)
    classifier = classifier_func(train_X, train_y, classifier_params)
    print("---------- classifier training ends ----------\n")

    #   分类器输出
    classifier_file = classifier_name
    classifier_params = sorted(classifier_params.items(), key=lambda x: x[0], reverse=False)
    for param, value in classifier_params:
        classifier_file += '_%s_%s' % (param, str(value))
    classifier_file = '%s/%s.h5' % (classifier_dst_dir, classifier_file)
    with open(classifier_file, 'wb') as rf:
        pickle.dump(classifier, rf, protocol=pickle.HIGHEST_PROTOCOL)
