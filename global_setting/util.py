ori_train_set_src_path = "../dataset/train.csv"
ori_test_set_src_path = "../dataset/test.csv"

multiple_cate_features = [
    'admission_type', 'insurance', 'marital_status', 'race'
]
binary_cate_features = [
    'ICU', 'NICU'
]
cate_features = multiple_cate_features + binary_cate_features
list_features = [
    'ICD9'
]
num_features = [
    'temperature', 'heartrate', 'resprate', 'o2sat',
    'sbp', 'dbp'
]
ori_features = cate_features + list_features + num_features
label = 'hospital_expire_flag'
