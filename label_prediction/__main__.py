#coding=utf-8
#coding=gbk
import json

import pandas as pd
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


def label_prediction_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regressor', help='the regressor to be used to predict test_Y', type=str)
    return parser


#   参数处理
args = label_prediction_parser().parse_args()
regressor_name = args.regressor

#   数据读取
test_X, test_y = pd.read_csv(test_X_src_path), pd.read_csv(test_set_src_path)[label]
test_X, test_y = test_X.to_numpy(), test_y.to_numpy()

#   回归器读取
regressor_params = sorted(regressor_params_map[regressor_name].items(), key=lambda x: x[0], reverse=False)
regressor_file_name = regressor_name
for param, value in regressor_params:
    regressor_file_name += '_%s_%s' % (param, str(value))
regressor_file = '%s/%s.h5' % (regressor_src_dir, regressor_file_name)
try:
    with open(regressor_file, 'rb') as rf:
        regressor = pickle.load(rf)
except Exception as e:
    sys.stderr(repr(e))
    exit()

#   结果预测
predict_y = regressor.predict(test_X)

#   结果输出
predict_y_df = pd.DataFrame({'LOS': predict_y})
predict_y_dst_path = predict_y_dst_dir + '/predict_y_%s.csv' % regressor_file_name
predict_y_df.to_csv(predict_y_dst_path, header=['LOS'], index=False)

#   评价结果输出
evaluation_result = cal_metric(test_y, predict_y)
evaluation_result_json = json.dumps(evaluation_result, indent=4)
print(evaluation_result_json)
evaluation_result_dst_path = evaluation_result_dst_dir + '/evaluation_result_%s.json' % regressor_file_name
with open(evaluation_result_dst_path, 'w') as er:
    er.write(evaluation_result_json)
