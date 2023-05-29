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
    parser.add_argument('--classifier', help='the classifier to be used to predict test_Y', type=str)
    return parser


#   参数处理
args = label_prediction_parser().parse_args()
classifier_name = args.classifier

#   数据读取
test_X, test_y = pd.read_csv(test_X_src_path), pd.read_csv(test_set_src_path)[label]
test_X, test_y = test_X.to_numpy(), test_y.to_numpy()

classifier_params_lst = classifier_params_lst_map[classifier_name]
for classifier_params in classifier_params_lst:
    classifier_params = sorted(classifier_params.items(), key=lambda x: x[0], reverse=False)
    #   分类器读取
    classifier_file_name = classifier_name
    for param, value in classifier_params:
        classifier_file_name += '_%s_%s' % (param, str(value))
    classifier_file = '%s/%s.h5' % (classifier_src_dir, classifier_file_name)
    try:
        with open(classifier_file, 'rb') as rf:
            classifier = pickle.load(rf)
    except Exception as e:
        sys.stderr(repr(e))
        exit()

    #   结果预测
    print("---------- result prediction starts ----------")
    print(classifier_params)
    pro_predict_y = classifier.predict_proba(test_X)
    predict_y = np.argmax(pro_predict_y, axis=1)
    print("---------- result prediction ends ----------\n")

    #   结果输出
    predict_y_df = pd.DataFrame({label: predict_y})
    predict_y_dst_path = predict_y_dst_dir + '/predict_y_%s.csv' % classifier_file_name
    predict_y_df.to_csv(predict_y_dst_path, header=[label], index=False)

    #   评价结果输出
    evaluation_result = cal_metric(test_y, predict_y, pro_predict_y)
    evaluation_result_json = json.dumps(evaluation_result, indent=4)
    print(evaluation_result_json + "\n")
    evaluation_result_dst_path = evaluation_result_dst_dir + '/evaluation_result_%s.json' % classifier_file_name
    with open(evaluation_result_dst_path, 'w') as er:
        er.write(evaluation_result_json)
