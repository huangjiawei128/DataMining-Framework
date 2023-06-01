train_X_src_path = "./data_preprocessing/output/train_X.csv"
test_X_src_path = "./data_preprocessing/output/test_X.csv"

train_X_dst_path = "./feature_expanding/output/train_X.csv"
test_X_dst_path = "./feature_expanding/output/test_X.csv"

window_size = 7

expand_feature_types = {
    'median': None, 'mean': None, 'standard_deviation': None,
    'variance': None, 'variance_larger_than_standard_deviation': None, 'variation_coefficient': None,
    'variation_coefficient': None, 'maximum': None, 'minimum': None,
    'mean_abs_change': None, 'mean_change': None, 'root_mean_square': None,
    'skewness': None, 'sample_entropy': None, 'mean_second_derivative_central': None
}

