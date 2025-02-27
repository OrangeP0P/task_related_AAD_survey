import numpy as np
import scipy.io as scio
from random import shuffle
from scipy.signal import resample

def read_data(fold_num, folder, subject_ID):
    sdn = 'data_' + subject_ID
    sdp = folder + sdn + '.mat'
    sln = 'label_' + subject_ID
    slp = folder + sln + '.mat'
    stn = 'trial_indices_' + subject_ID
    stp = folder + stn + '.mat'

    # 读取 源域、目标域 数据集和标签
    all_data = scio.loadmat(sdp)[sdn]
    all_label = scio.loadmat(slp)[sln]
    all_trial_indices = scio.loadmat(stp)[stn]

    ################################################# 数据集降采样 ######################################################
    all_data = all_data[:, :, 0:64]
    s0, s1, s3 = np.shape(all_data)
    re_s1 = int(s1 / 4)
    all_data = all_data[:, 0:4 * re_s1, :]

    data = np.transpose(all_data, (2, 1, 0))
    data = np.reshape(data, (64, -1))

    resampled_eeg_data = np.zeros((64, int(data.shape[1] / 4)))

    for channel_idx in range(data.shape[0]):
        resampled_eeg_data[channel_idx, :] = resample(data[channel_idx, :], num=int(data.shape[1] / 4))

    data = np.reshape(resampled_eeg_data, (64, int(s1 / 4), -1))
    all_data = np.transpose(data, (2, 1, 0))
    ##################################################################################################

    ##################################### 故事划分 #####################################################
    if fold_num < 1 or fold_num > 6:
        raise ValueError("fold_num 必须在1到6之间")

    # 获取所有的trial编号
    all_trials = np.unique(all_trial_indices[:, 0])

    # 每个fold的大小
    trials_per_fold = len(all_trials) // 6

    # 确定验证集和测试集的trial编号范围
    val_test_start = (fold_num - 1) * trials_per_fold
    val_test_end = val_test_start + trials_per_fold
    val_test_trials = all_trials[val_test_start:val_test_end]

    # 从剩余的trial中选取40个作为训练集
    remaining_trials = np.setdiff1d(all_trials, val_test_trials)
    train_trials = remaining_trials[:40]

    # 构建训练集、验证集和测试集的样本索引列表
    train_list = np.where(np.isin(all_trial_indices[:, 0], train_trials))[0]
    test_list = np.where(np.isin(all_trial_indices[:, 0], val_test_trials))[0]

    # print(f"训练集trial列表 train_trials: {train_trials}")
    # print(f"验证集trial列表 val_trials: {val_test_trials} 的每个trial的前一半样本")
    # print(f"测试集trial列表 test_trials: {val_test_trials} 的每个trial的后一半样本")
    ##################################################################################################

    ########################################## 数据集制作 ##############################################
    tr_num = np.size(train_list)  # 训练集样本数目
    te_num = np.size(test_list)  # 测试集样本数目

    # 构建训练集、验证集、测试集 数据和标签
    train_indices = np.array(train_list)
    test_indices = np.array(test_list)
    train_data = all_data[train_indices, :, :]
    _train_label = all_label[train_indices, :]
    test_data = all_data[test_indices, :, :]
    _test_label = all_label[test_indices, :]

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
    ##################################################################################################

    return train_data, train_label, test_data, test_label, tr_num, te_num