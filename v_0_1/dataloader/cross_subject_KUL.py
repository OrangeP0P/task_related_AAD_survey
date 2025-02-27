import numpy as np
import scipy.io as scio
from random import shuffle
from scipy.signal import resample

def read_data(path, subject_ID):

    sdn = 'data_' + subject_ID
    sdp = path + sdn + '.mat'
    sln = 'label_' + subject_ID
    slp = path + sln + '.mat'

    # 读取 源域、目标域 数据集和标签
    all_data = scio.loadmat(sdp)[sdn]
    all_label = scio.loadmat(slp)[sln]
    all_data = all_data[:, :, 0:64]

    # 计算样本量，训练集、测试集数目
    n_sample = np.size(all_data, 0)
    datasets_list = list(range(0, n_sample))

    shuffle(datasets_list)  # 改变后的数据集编号

    # 构建训练集、验证集、测试集 数据和标签
    train_data = all_data
    _train_label = all_label

    train_label = np.array(range(0, len(_train_label)))
    for i in range(0, len(_train_label)):
        train_label[i] = _train_label[i]

    train_label = train_label - 1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))

    return train_data, train_label, n_sample

def read_source_data(folder, source_subject_ID):

    sdn = 'data_' + source_subject_ID
    sdp = folder + sdn + '.mat'
    sln = 'label_' + source_subject_ID
    slp = folder + sln + '.mat'

    # 读取 源域、目标域 数据集和标签
    all_data = scio.loadmat(sdp)[sdn]
    all_label = scio.loadmat(slp)[sln]

    # 计算样本量，训练集、测试集数目
    n_sample = np.size(all_data, 0)
    datasets_list = list(range(0, n_sample))
    shuffle(datasets_list)  # 改变后的数据集编号

    # 构建训练集、验证集、测试集 数据和标签
    source_data = all_data[datasets_list, :, :]
    _source_label = all_label[datasets_list, :]

    source_label = np.array(range(0, len(_source_label)))
    for i in range(0, len(_source_label)):
        source_label[i] = _source_label[i]
    source_label = source_label - 1
    source_data = np.transpose(np.expand_dims(source_data, axis=1), (0, 1, 3, 2))
    return source_data, source_label