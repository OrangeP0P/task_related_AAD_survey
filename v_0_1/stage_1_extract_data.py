import numpy as np

'''

Note that for different classification tasks, the output (1) data shape, and (2) label shape are the same, so you can
use only one model class to finish all the four classification tasks.

'''

# =================================== I. For Within-Subject Experiment ===================================

'''
You can choose within_subject dataloader from different datasets via:

--------------------------------
"dataloader.within_subject_XXX"

such as: 
(1) within_subject_KUL, 
(2) within_subject_DTU, 
(3) within_subject_UHD
--------------------------------

Below is an example of loading cross-subject data from KUL dataset

[Input]
    subject_ID_test = int → Set the subject ID for testing (ranging from 1~n, n is the total number of subject)
    subject_ID_list = list → subject list of dataset
    data_path = str → your data path

[Output]
    train_data = ndarray → 4D ndarray, Shape: [sample_num, 1, electrode_num, sampling_rate × window_size]
    train_label = ndarray → 1D ndarray, Shape: [sample_num, ], 0 indicate left, 1 indicate right
    train_num = int → The sample number of the training set

    val_data = ndarray → 4D ndarray, Shape: [sample_num, 1, electrode_num, sampling_rate × window_size]
    val_label = ndarray → 1D ndarray, Shape: [sample_num, ], 0 indicate left, 1 indicate right
    val_num = int → The sample number of the validation set

    test_data = ndarray → 4D ndarray, Shape: [sample_num, 1, electrode_num, sampling_rate × window_size]
    test_label = ndarray → 1D ndarray, Shape: [sample_num, ], 0 indicate left, 1 indicate right
    test_num = int → The sample number of the testing set

    **Note**: 
            The ratio of train-validation-test is 3-1-1*
'''

from dataloader.within_subject_KUL import read_data

# ------------------------------ Setting of the Dataloader -----------------------------------
fold_num = 1  # Set the cross-validation number (ranging from 1~5)
subject_ID_test = 1  # Set the subject ID for testing (ranging from 1~n, n is the total number of subject in dataset)
subject_ID_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                   '11', '12', '13', '14', '15', '16', '17', '18']  # subject list of DTU dataset
data_path = r'../Data/KUL/1s/'  # data path of your data

# ------------------------------ Load the training and testing data --------------------------------------
train_data, train_label, validation_data, validation_label, test_data, test_label, train_num, val_num, test_num \
    = read_data(fold_num, data_path, str(subject_ID_test))



# =================================== II. For Cross-Subject Experiment =========================================

'''
You can choose cross_subject dataloader from different datasets via:

--------------------------------
"dataloader.cross_subject_XXX"

such as: 
(1) cross_subject_KUL, 
(2) cross_subject_DTU, 
(3) cross_subject_UHD
--------------------------------

Below is an example of loading cross-subject data from DTU dataset

[Input]
    subject_ID_test = int → Set the subject ID for testing (ranging from 1~n, n is the total number of subject)
    subject_ID_list = list → subject list of dataset
    data_path = str → your data path
    
[Output]
    train_data = ndarray → 4D ndarray, Shape: [sample_num, 1, electrode_num, sampling_rate × window_size]
    train_label = ndarray → 1D ndarray, Shape: [sample_num, ], 0 indicate left, 1 indicate right
    train_num = int → The sample number of the training set

    test_data = ndarray → 4D ndarray, Shape: [sample_num, 1, electrode_num, sampling_rate × window_size]
    test_label = ndarray → 1D ndarray, Shape: [sample_num, ], 0 indicate left, 1 indicate right
    test_num = int → The sample number of the testing set

    
'''

from dataloader.cross_subject_DTU import read_data, read_source_data

# ------------------------------ Setting of the Dataloader -----------------------------------
subject_ID_test = 1  # Set the subject ID for testing (ranging from 1~n, n is the total number of subject in dataset)
subject_ID_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                   '11', '12', '13', '14', '15', '16', '17', '18']  # subject list of DTU dataset
data_path = r'../Data/DTU/1s/'  # data path of your data

# ------------------------------ Load the training data --------------------------------------
source_subject_ID_list = [id for id in subject_ID_list if id != subject_ID_list[subject_ID_test - 1]]
train_data = np.empty((0, 1, 64, 128))
source_label = np.empty((0,))  # label of all the source data
subject_ID_test = subject_ID_list[subject_ID_test - 1]  # select subject
print('Training Subject ID = ', subject_ID_test)

for source_subject_ID in source_subject_ID_list:
    print('Loading Testing Subject ID = ', source_subject_ID)
    subject_data, subject_label = read_source_data(data_path, source_subject_ID)
    train_data = np.concatenate((train_data, subject_data), axis=0)
    source_label = np.concatenate((source_label, subject_label), axis=0)

train_num = np.size(train_data, 0)  # The sample number of the training set

# ----------------------------- Load the testing data ---------------------------------------
test_data, test_label, test_num = (read_data(data_path, subject_ID_test))


# =================================== III. For Multi-Modal Experiment =======================================

'''
You can choose within_subject dataloader from different datasets via:

--------------------------------
"dataloader.multi_modal_XXX"

such as: 
(1) multi_modal_KUL, 
(2) multi_modal_DTU
--------------------------------

Below is an example of loading multi-modal data from KUL dataset

[Input]
    subject_ID_test = int → Set the subject ID for testing (ranging from 1~n, n is the total number of subject)
    subject_ID_list = list → subject list of dataset
    data_path = str → your data path

[Output]
    train_data = ndarray → 4D ndarray, Shape: [sample_num, 1, electrode_num, sampling_rate × window_size]
    train_label = ndarray → 1D ndarray, Shape: [sample_num, ], 0 indicate left, 1 indicate right
    train_num = int → The sample number of the training set

    val_data = ndarray → 4D ndarray, Shape: [sample_num, 1, electrode_num, sampling_rate × window_size]
    val_label = ndarray → 1D ndarray, Shape: [sample_num, ], 0 indicate left, 1 indicate right
    val_num = int → The sample number of the validation set

    test_data = ndarray → 4D ndarray, Shape: [sample_num, 1, electrode_num, sampling_rate × window_size]
    test_label = ndarray → 1D ndarray, Shape: [sample_num, ], 0 indicate left, 1 indicate right
    test_num = int → The sample number of the testing set

    **Note**: 
            (1) The ratio of train-validation-test is 3-1-1
            (2) The last two channel is the processed left and right audio envelopes, you can use them for multi
                modal research
'''

from dataloader.multi_modal_KUL import read_data

# ------------------------------ Setting of the Dataloader -----------------------------------
fold_num = 1  # Set the cross-validation number (ranging from 1~5)
subject_ID_test = 1  # Set the subject ID for testing (ranging from 1~n, n is the total number of subject in dataset)
data_path = r'../Data/KUL/1s/'  # data path of your data

# ------------------------------ Load the training and testing data --------------------------------------
train_data, train_label, validation_data, validation_label, test_data, test_label, train_num, val_num, test_num \
    = read_data(fold_num, data_path, str(subject_ID_test))


# =================================== IV. For Cross-Trial Experiment =======================================

'''
You can choose cross_trial dataloader from different datasets via:

--------------------------------
"dataloader.cross_trial_XXX"

such as: 
(1) cross_trial_DTU, 
(2) cross_trial_KUL,
(3) cross_trial_UHD
--------------------------------

Note that for each dataset, the trial numbers and data setting is totally different, so it is important to use the
correct dataloader for the corresponding dataset.

Below is an example of loading multi-modal data from KUL dataset

[Input]
    subject_ID_test = int → Set the subject ID for testing (ranging from 1~n, n is the total number of subject)
    subject_ID_list = list → subject list of dataset
    data_path = str → your data path

[Output]
    train_data = ndarray → 4D ndarray, Shape: [sample_num, 1, electrode_num, sampling_rate × window_size]
    train_label = ndarray → 1D ndarray, Shape: [sample_num, ], 0 indicate left, 1 indicate right
    train_num = int → The sample number of the training set

    val_data = ndarray → 4D ndarray, Shape: [sample_num, 1, electrode_num, sampling_rate × window_size]
    val_label = ndarray → 1D ndarray, Shape: [sample_num, ], 0 indicate left, 1 indicate right
    val_num = int → The sample number of the validation set

    test_data = ndarray → 4D ndarray, Shape: [sample_num, 1, electrode_num, sampling_rate × window_size]
    test_label = ndarray → 1D ndarray, Shape: [sample_num, ], 0 indicate left, 1 indicate right
    test_num = int → The sample number of the testing set


    **Note**: 
            (1) The ratio of train-validation-test is 3-1-1
            (2) The last two channel is the processed left and right audio envelopes, you can use them for multi
                modal research
'''

from dataloader.cross_trial_KUL import read_data

# ------------------------------ Setting of the Dataloader -----------------------------------
fold_num = 1  # Set the cross-validation number (ranging from 1~5)
subject_ID_test = 1  # Set the subject ID for testing (ranging from 1~n, n is the total number of subject in dataset)
data_path = r'../Data/KUL/1s/'  # data path of your data

# ------------------------------ Load the training and testing data --------------------------------------
train_data, train_label, test_data, test_label, train_num, test_num \
    = read_data(fold_num, data_path, str(subject_ID_test))