import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from model.SNN_module import *
import torch.utils.data

class RecSNN(nn.Module):
    def __init__(self):
        super(RecSNN, self).__init__()
        self.channel = 64
        self.classes = 2
        self.n_filters = 5

        self.SNN1 = nn.Sequential()
        self.SNN1.add_module('linear1', nn.Linear(64, 64))
        self.SNN1.add_module('snn-rec', SNNRecLayer(hid_size=64))

        self.Readout = nn.Sequential()
        self.Readout.add_module('linear2', nn.Linear(64, 5))
        self.Readout.add_module('bn', nn.BatchNorm1d(5, False))
        self.Readout.add_module('linear3', nn.Linear(5, 2))

    def forward(self, source_data):
        source_data = source_data.squeeze()
        source_data = source_data.permute(2, 0, 1).contiguous()
        feature_source = self.SNN1(source_data)
        output = self.Readout(feature_source)
        return output


class ShallowCNN(nn.Module):  # Frame1: 非常浅的卷积层
    def __init__(self):
        super(ShallowCNN, self).__init__()
        self.channel = 64
        self.classes = 2
        self.n_filters = 5

        self.Block1 = nn.Sequential()
        self.Block1.add_module('b1-1', nn.Conv2d(1, 5, (self.channel, 17), padding=0))
        self.Block1.add_module('b1-3', nn.ReLU())
        self.Block1.add_module('b1-2', nn.AvgPool2d(kernel_size=(1, 112)))
        # self.Block1.add_module('b1-3', nn.ReLU())

        self.Classifier = nn.Sequential()
        self.Classifier.add_module('fc-1', nn.Linear(5, 5))
        self.Classifier.add_module('b-1', nn.BatchNorm1d(5, False))
        self.Classifier.add_module('fc-2', nn.Linear(5, 2))
        # self.Classifier.add_module('fc-2', nn.ReLU())
        # self.Classifier.add_module('fc-3', nn.Linear(5, 2))

    def forward(self, input_data):
        feature_source = self.Block1(input_data)
        feature_source = feature_source.view(-1, 5)  # 展开为一维
        output = self.Classifier(feature_source)  # 分类结果
        return output
class depthwise_separable_conv(nn.Module):  # 深度可分离卷积
    def __init__(self, nin, nout, kernel_size):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nout, kernel_size=(1, kernel_size), padding=0, groups=nin)
        self.pointwise = nn.Conv2d(nout, nout, kernel_size=1)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.Kernel = 40
        self.F1 = 16
        self.DF = 10
        self.Channel = 64
        self.Class = 2
        self.mapsize = 16 * 160

        self.Extractor = nn.Sequential()
        self.Extractor.add_module('c-1', nn.Conv2d(1, self.F1, (1, self.Kernel), padding=0))
        self.Extractor.add_module('p-1', nn.ZeroPad2d((int(self.Kernel / 2) - 1, int(self.Kernel / 2), 0, 0)))
        self.Extractor.add_module('b-1', nn.BatchNorm2d(self.F1, False))

        self.Extractor.add_module('c-2', nn.Conv2d(self.F1, self.F1 * self.DF, (self.Channel, 1), groups=8))
        self.Extractor.add_module('b-2', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-1', nn.ELU())

        self.Extractor.add_module('a-1', nn.AvgPool2d(kernel_size=(1, 2)))
        self.Extractor.add_module('d-1', nn.Dropout(p=0.25))

        self.Extractor.add_module('c-3', depthwise_separable_conv(self.F1 * self.DF, self.F1 * self.DF, int(self.Kernel / 4)))
        self.Extractor.add_module('p-2', nn.ZeroPad2d((int(self.Kernel / 8) - 1, int(self.Kernel / 8), 0, 0)))
        self.Extractor.add_module('b-3', nn.BatchNorm2d(self.F1 * self.DF, False))
        self.Extractor.add_module('e-2', nn.ELU())
        self.Extractor.add_module('a-2', nn.AvgPool2d(kernel_size=(1, 4)))
        self.Extractor.add_module('d-2', nn.Dropout(p=0.25))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc-1', nn.Linear(self.mapsize, 1024))
        # self.class_classifier.add_module('fcb-1', nn.BatchNorm1d(1024, False))
        self.class_classifier.add_module('fc-2', nn.Linear(1024, 512))
        # self.class_classifier.add_module('fcb-2', nn.BatchNorm1d(512, False))
        self.class_classifier.add_module('fc-3', nn.Linear(512, 256))
        # self.class_classifier.add_module('fcb-3', nn.BatchNorm1d(256, False))
        self.class_classifier.add_module('fc-4', nn.Linear(256, 64))
        # self.class_classifier.add_module('fcb-4', nn.BatchNorm1d(64, False))
        self.class_classifier.add_module('fc-5', nn.Linear(64, 2))

    def forward(self, source_data):
        feature = self.Extractor(source_data)
        _, s2, _, s4 = feature.shape
        feature = feature.view(-1, s2*s4)
        class_output = self.class_classifier(feature)
        return class_output

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.channel = 64
        self.classes = 2
        self.n_filters = 5
        self.linear1 = nn.Linear(64, 64)
        self.lstm = nn.LSTM(64, 64, num_layers=1, batch_first=True)

        self.Readout = nn.Sequential()
        self.Readout.add_module('linear2', nn.Linear(64, 5))
        self.Readout.add_module('bn', nn.BatchNorm1d(5, False))
        self.Readout.add_module('linear3', nn.Linear(5, 2))

    def forward(self, input_data):
        input_data = input_data.squeeze()
        input_data = input_data.permute(2, 0, 1).contiguous()
        input_data = self.linear1(input_data)
        h0 = torch.zeros(1, input_data.size(0), 64).to(input_data.device)
        c0 = torch.zeros(1, input_data.size(0), 64).to(input_data.device)
        feature_source, _ = self.lstm(input_data, (h0, c0))
        output = self.Readout(feature_source.mean(0))
        return output

class CNN_LSTM(nn.Module):
    def __init__(self,
                 in_chans,
                 n_classes,
                 input_time_length=128,
                 crop_length=64,

                 n_filters_time=40,
                 time_size=25,
                 n_filters_spat=30,

                 poolmax_size=20,
                 poolmax_stride=5,

                 batch_norm=True,
                 batch_norm_alpha=0.1,
                 drop_prob1=0.5, ):

        super(CNN_LSTM, self).__init__()
        self.__dict__.update(locals())

        del self.self

        self.conv_time = nn.Conv2d(1, self.n_filters_time, (self.time_size, 1), stride=(1, 1), bias=False, )
        self.conv_ica = nn.Conv2d(self.n_filters_time, self.n_filters_spat, (1, self.in_chans), stride=(1, 1),
                                  bias=False, )

        self.batch1 = nn.BatchNorm2d(self.n_filters_spat, momentum=self.batch_norm_alpha, affine=True, eps=1e-5, )
        self.n_crop = int(self.crop_length - self.time_size + 1 / 5 - 3)

        self.poolmean = nn.AvgPool2d(kernel_size=(self.poolmax_size, 1), stride=(self.poolmax_stride, 1))
        self.lstm = nn.LSTM(self.n_filters_spat, self.n_filters_spat, 1)

        self.dropout1 = nn.Dropout(drop_prob1)
        self.dropout2 = nn.Dropout(drop_prob1)
        self.dropout3 = nn.Dropout(drop_prob1)

        self.conv_class = nn.Conv2d(1,
                                    self.n_classes,
                                    kernel_size=(17, self.n_filters_spat),
                                    stride=(1, 1),
                                    bias=True)

        self.softmax = nn.LogSoftmax(dim=1)

        nn.init.xavier_uniform_(self.conv_ica.weight, gain=1)
        nn.init.xavier_uniform_(self.conv_time.weight, gain=1)

        nn.init.constant_(self.batch1.weight, 1)
        nn.init.constant_(self.batch1.bias, 0)

        nn.init.xavier_uniform_(self.conv_class.weight, gain=1)
        nn.init.constant_(self.conv_class.bias, 0)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)

        x = self.conv_time(x)
        x = self.conv_ica(x)

        x = self.batch1(x)
        x = self.dropout1(x)

        x = torch.mul(x, x)
        x = self.poolmean(x)
        x = x.squeeze()
        x = x.permute(0, 2, 1)
        x, (h, c) = self.lstm(x, None)
        x = self.dropout2(x)
        x = x.unsqueeze(1)

        x = self.softmax(self.conv_class(x))
        x = x.squeeze()

        return x

def choose_act_func(act_name):
    if act_name == 'elu':
        return nn.ELU()
    elif act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'lrelu':
        return nn.LeakyReLU()
    else:
        raise TypeError('activation_function type not defined.')

def handle_param(args, net):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'rmsp':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)
    else:
        raise TypeError('optimizer type not defined.')
    if args.loss_function == 'CrossEntropy':
        loss_function = nn.CrossEntropyLoss()
    elif args.loss_function == 'BCELoss':
        loss_function = nn.BCELoss()
    else:
        raise TypeError('loss_function type not defined.')
    return optimizer, loss_function

'''选取网络和激活函数'''
def choose_net(args):
    if args.model == 'ShallowCNN':
        return {
        'elu': [ShallowCNN()]
        }
    elif args.model == 'DeepCNN':
        return {
        'elu': [DeepCNN()]
        }
    elif args.model == 'LSTM':
        return {
        'elu': [LSTM()]
        }
    elif args.model == 'RecSNN':
        return {
        'elu': [RecSNN()]
        }
    elif args.model == 'CNN_LSTM':
        return {
        'elu': [CNN_LSTM(64,2)]
        }
    else:
        raise TypeError('model type not defined.')