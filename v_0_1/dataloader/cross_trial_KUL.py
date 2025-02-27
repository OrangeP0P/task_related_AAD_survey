import numpy as np
import scipy.io as scio
from random import shuffle
from scipy.signal import resample

def read_data(fold_num, folder, subject_ID):
    sdn = 'data_' + subject_ID
    sdp = folder + sdn + '.mat'
    sln = 'label_' + subject_ID
    slp = folder + sln + '.mat'

    # 读取 源域、目标域 数据集和标签
    all_data = scio.loadmat(sdp)[sdn]
    all_label = scio.loadmat(slp)[sln]
    all_data = all_data[:, :, 0:64]

    # 计算样本量，训练集、测试集数目
    n_sample = np.size(all_data, 0)
    datasets_list = list(range(0, n_sample))

    trial_indices = []
    start_index = 0
    current_label = all_label[0]

    train_trials = 6  # 训练集 trial 数量
    test_trials = 2  # 测试集 trial 数量

    for i in range(1, len(all_label)):
        if all_label[i] != current_label:
            trial_indices.append((start_index, i - 1))
            start_index = i
            current_label = all_label[i]

    trial_indices.append((start_index, len(all_label) - 1))  # 加入最后一个trial
    trial_indices = trial_indices[:8]  # 只保留前 8 个 trial

    # Select trials for training and testing based on fold_num
    test_trial_start = (fold_num - 1) * test_trials
    test_trial_end = test_trial_start + test_trials
    train_trials_list = [i for i in range(len(trial_indices)) if i not in range(test_trial_start, test_trial_end)]

    tr_start = trial_indices[train_trials_list[0]][0]
    tr_end = trial_indices[train_trials_list[-1]][1]
    te_start = trial_indices[test_trial_start][0]
    te_end = trial_indices[test_trial_end - 1][1]
    tr_num = tr_end - tr_start
    te_num = te_end - te_start

    # 构建训练集、验证集、测试集 数据和标签
    train_data = all_data[datasets_list[tr_start:tr_end], :, :]
    _train_label = all_label[datasets_list[tr_start:tr_end], :]
    test_data = all_data[datasets_list[te_start:te_end], :, :]
    _test_label = all_label[datasets_list[te_start:te_end], :]

    train_label = np.array(range(0, len(_train_label)))
    for i in range(0, len(_train_label)):
        train_label[i] = _train_label[i]
    test_label = np.array(range(0, len(_test_label)))
    for i in range(0, len(_test_label)):
        test_label[i] = _test_label[i]

    train_label = train_label - 1
    test_label = test_label - 1

    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    return train_data, train_label, test_data, test_label, tr_num, te_num