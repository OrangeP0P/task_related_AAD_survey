'''
Source code of AAD survey paper:
Author: CHEN Zhige

Once you load data using the dataloader, you can start experiment by first choose the model.
In model class, there are five baseline models for you to choose.
Note that these model is just the baseline, and the sota approaches will have a better performance (around 5%~8%)
I will provide you more sota models in AAD decoding later.

** Note **
    (1) This demo only use the data from fold-1, you need to construct iteration to apply 5-fold cross validations
'''

import numpy as np
import torch
import torch.utils.data as Data
from model.model import choose_net, handle_param

# =============================================== Set parameters ================================================
Test_Name = 'ShallowCNN'  # You can choose different models: DeepCNN, LSTM, CNN_LSTM, RecSNN
Batch_size = 256
Epoch = 100  # Set epoch numbers

# ===================================== Dataloader We have defined in Stage 1 ====================================
from dataloader.multi_modal_KUL import read_data
fold_num = 1  # Set the cross-validation number (ranging from 1~5)
subject_ID_test = '1'  # Set the subject ID for testing (ranging from 1~n, n is the total number of subject in dataset)
subject_ID_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                   '11', '12', '13', '14', '15', '16', '17', '18']  # subject list of DTU dataset
data_path = r'../Data/KUL/1s/'  # data path of your data

train_data, train_label, validation_data, validation_label, test_data, test_label, train_num, val_num, test_num \
    = read_data(fold_num, data_path, subject_ID_test)

# ==================================== Model Training ============================================================
class Args:
    def __init__(self):
        self.model = Test_Name
        self.activation_function = 'relu'
        self.optimizer = 'adam'
        self.loss_function = 'CrossEntropy'
        self.learning_rate = 1e-3
args = Args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_loader(data, label, batch_size, shuffle=True):
    dataset = Data.TensorDataset(
        torch.from_numpy(data.astype(np.float32)),
        torch.from_numpy(label.astype(np.float32))
    )
    return Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = create_loader(train_data, train_label, Batch_size)
val_loader = create_loader(validation_data, validation_label, Batch_size, False)
test_loader = create_loader(test_data, test_label, Batch_size, False)

net_dict = choose_net(args)
model = list(net_dict.values())[0][0].to(device)

optimizer, criterion = handle_param(args, model)

def train_epoch():
    model.train()
    total_loss, correct = 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device).long()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
    return total_loss / len(train_loader), correct / len(train_loader.dataset)

def evaluate(loader):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device).long()
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

best_model_path = f'best_{Test_Name}_model.pth'

print(f"Training {Test_Name} on subject {subject_ID_test}...")
best_acc = 0.0

for epoch in range(1, Epoch + 1):
    train_loss, train_acc = train_epoch()
    val_loss, val_acc = evaluate(val_loader)
    test_loss, test_acc = evaluate(test_loader)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved at epoch {epoch}!")

    print(f"Epoch {epoch:02d} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | "
          f"Test Acc: {test_acc:.2%}")

print("\n Testing best model...")
model.load_state_dict(torch.load(best_model_path))
model.eval()

final_test_loss, final_test_acc = evaluate(test_loader)
print(f"Final Test Accuracy using best model: {final_test_acc:.2%}")
print(f"Best Validation Accuracy: {best_acc:.2%}")