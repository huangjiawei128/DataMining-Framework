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
    parser.add_argument('--classifier', help='the classifier to be trained', type=str)
    return parser


#   参数处理
args = model_building_parser().parse_args()
classifier_name = args.classifier

#   数据读取
train_X, train_y = pd.read_csv(train_X_src_path), pd.read_csv(train_y_src_path)
train_X, train_y = train_X.to_numpy(), train_y.to_numpy()

#   分类器训练
classifier_params = classifier_params_map[classifier_name]
classifier = classifier_map[classifier_name](train_X, train_y, classifier_params)

#   分类器输出
classifier_file = classifier_name
classifier_params = sorted(classifier_params.items(), key=lambda x: x[0], reverse=False)
for param, value in classifier_params:
    classifier_file += '_%s_%s' % (param, str(value))
classifier_file = '%s/%s.h5' % (classifier_dst_dir, classifier_file)
print(classifier_file)
with open(classifier_file, 'wb') as rf:
    pickle.dump(classifier, rf, protocol=pickle.HIGHEST_PROTOCOL)
