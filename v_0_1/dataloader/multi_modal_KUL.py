import numpy as np
import scipy.io as scio
from random import shuffle
from scipy.signal import resample

def read_data(fold_num, path, subject_ID):

    sdn = 'data_' + subject_ID
    sdp = path + sdn + '.mat'
    sln = 'label_' + subject_ID
    slp = path + sln + '.mat'

    # 读取 源域、目标域 数据集和标签
    all_data = scio.loadmat(sdp)[sdn]
    all_label = scio.loadmat(slp)[sln]
    all_data = all_data[:, :, 0:66]

    # 计算样本量，训练集、测试集数目
    n_sample = np.size(all_data, 0)
    datasets_list = list(range(0, n_sample))

    shuffle(datasets_list)  # 改变后的数据集编号

    if fold_num == 1:
        te_folder = 1
        va_folder = 5
    else:
        va_folder = fold_num - 1
        te_folder = fold_num

    one_fold_num = round(n_sample/5)
    tr_num = round((n_sample/5) * 3)  # 训练集数据大小
    va_num = round((n_sample/5) * 1)  # 验证集数据大小
    te_num = round((n_sample/5) * 1)  # 测试集数据大小

    te_start = (te_folder-1) * one_fold_num
    va_start = (va_folder-1) * one_fold_num

    test_list = datasets_list[te_start:te_start+te_num]
    validation_list = datasets_list[va_start:va_start+va_num]
    exclude = set(validation_list + test_list)
    train_list = [x for x in datasets_list if x not in exclude]

    shuffle(test_list)
    shuffle(validation_list)
    shuffle(train_list)

    # 构建训练集、验证集、测试集 数据和标签
    train_data = all_data[train_list, :, :]
    _train_label = all_label[train_list, :]
    validation_data = all_data[validation_list, :, :]
    _validation_label = all_label[validation_list, :]
    test_data = all_data[test_list, :, :]
    _test_label = all_label[test_list, :]

    train_label = np.array(range(0, len(_train_label)))
    for i in range(0, len(_train_label)):
        train_label[i] = _train_label[i]
    validation_label = np.array(range(0, len(_validation_label)))
    for i in range(0, len(_validation_label)):
        validation_label[i] = _validation_label[i]
    test_label = np.array(range(0, len(_test_label)))
    for i in range(0, len(_test_label)):
        test_label[i] = _test_label[i]

    train_label = train_label - 1
    validation_label = validation_label - 1
    test_label = test_label - 1

    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    validation_data = np.transpose(np.expand_dims(validation_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    return train_data, train_label, validation_data, validation_label, test_data, test_label, tr_num, va_num, te_num