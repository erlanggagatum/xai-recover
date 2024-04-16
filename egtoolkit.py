import tensorflow as tf
import keras
from keras import layers
from keras import Model, Sequential
from keras.saving import load_model
from keras.layers import *
from keras.layers import LSTM, Dense, Input, Reshape, Dropout, TimeDistributed, Conv1D, MaxPool1D, Flatten
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import *
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import torch.nn.init as init

import statistics

import random

class NormalizeSequenceDataset:
    def __init__(self, train_sequence, val_sequence, test_sequence):
        self.train_sequence = train_sequence
        self.val_sequence = val_sequence
        self.test_sequence = test_sequence
        self.x_scaler = MinMaxScaler(feature_range=(0,1))
        self.y_scaler = MinMaxScaler(feature_range=(0,1))
        self.setup()
        
    def setup(self):
        # get all data
        list_x_train_dataset = np.array([df[0].values for df in self.train_sequence])
        
        # setup scaler for x
        self.x_scaler = self.x_scaler.fit(list_x_train_dataset.reshape(-1, list_x_train_dataset.shape[-1]))

        # setup scaler for y
        self.y_scaler.data_min_ = np.array([self.x_scaler.data_min_[0]])
        self.y_scaler.data_max_ = np.array([self.x_scaler.data_max_[0]])
        self.y_scaler.feature_range = (0, 1)
        
        # Compute the scale and min_ based on the feature_range and data_min_, data_max_
        self.y_scaler.scale_ = (self.y_scaler.feature_range[1] - self.y_scaler.feature_range[0]) / (self.y_scaler.data_max_ - self.y_scaler.data_min_)
        self.y_scaler.min_ = self.y_scaler.feature_range[0] - self.y_scaler.data_min_ * self.y_scaler.scale_
        
        print('done: setup normalization scaler..')
        
    def normalize_data(self):
        self.normalized_train_sequence = None
        self.normalized_val_sequence = None
        self.normalized_test_sequence = None
        
        # normalize train
        if self.train_sequence != []:
            self.normalized_train_sequence = self.perform_normalization(self.train_sequence)
        # normalize val
        if self.val_sequence != []:
            self.normalized_val_sequence = self.perform_normalization(self.val_sequence)
        # normalize test
        if self.test_sequence != []:
            self.normalized_test_sequence = self.perform_normalization(self.test_sequence)
        
        print('done: setup normalized train, val, and test sequence data')
        return self.normalized_train_sequence, self.normalized_val_sequence, self.normalized_test_sequence
        
        
    def perform_normalization(self, sequenced_data):
        list_x = np.array([df[0].values for df in sequenced_data])
        
        # print([df[1] for df in sequenced_data])
        
        y = np.array([df[1] for df in sequenced_data])
        # print(y.shape, list_x.shape)


        list_y = np.array([df[1].values for df in sequenced_data])
        
        # normalize x
        norm_x = self.x_scaler.transform(list_x.reshape(-1, list_x.shape[-1])).reshape(list_x.shape)
        # norm_y = self.y_scaler.transform(list_y)
        norm_y = self.y_scaler.transform(list_y.reshape(-1, list_y.shape[-1])).reshape(list_y.shape)
        
        sequences = []
        
        for _, (x,y) in enumerate(zip(norm_x, norm_y)):
            sequences.append((x, y))
            
        return sequences

# Power Consumption Dataset Class
class PCDataset(Dataset):
    def __init__(self, sequences):
        # super().__init__()
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return dict(
            sequence = torch.Tensor(sequence).to(device='cuda') if torch.cuda.is_available() else torch.Tensor(sequence.to_numpy()), # put to cuda device if available
            label = torch.Tensor(label).float().to(device='cuda') if torch.cuda.is_available() else torch.Tensor(label).float() # put to cuda device if available
        )
        

class PCDataModule():
    def __init__(self, train_sequences, test_sequences, val_sequences, batch_size=8):
        print('initialize data module..')
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size
        # print('batch size', self.batch_size)
        
    def setup(self, stage=None):
        self.train_dataset = PCDataset(self.train_sequences)
        self.val_dataset = PCDataset(self.val_sequences)
        self.test_dataset = PCDataset(self.test_sequences)
        print('done..')
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            # num_workers=2
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            # num_workers=2
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            # num_workers=2
        )

def scaler(dataset, mode='minmax'):
    
    dataset_norm = np.copy(dataset)
    if mode == 'minmax': # default min max scaller
        num_cols = dataset.shape[1]
        for c in range(num_cols):
            dataset_norm[:, c] -= dataset[:, c].min()
            dataset_norm[:, c] /= (dataset[:, c].max()-dataset[:, c].min())
            
    return dataset_norm

def pd_import(file, filetype='csv'):
    return pd.read_csv(file, 
                          low_memory=False,
                          delimiter=',',
                          parse_dates=[0],
                          dayfirst=True,
                          index_col='datetime')
    
def create_sequences(input_data: pd.DataFrame, target_column, sequence_length, offset = 1):
    if target_column not in input_data.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in the input data.")

    sequences = []
    data_size = len(input_data)

    for i in range(data_size - sequence_length - offset + 1):
        sequence = input_data.iloc[i:i + sequence_length]

        label_position = i + sequence_length
        label = input_data.iloc[label_position:label_position + offset][target_column]


        sequences.append((sequence, label))
    
    return sequences

# default
# def create_sequences(input_data: pd.DataFrame, target_column, sequence_length, offset = 1):
#     sequences = []
#     data_size = len(input_data)
    
#     for i in range(data_size - sequence_length):
#         sequence = input_data[i:i+sequence_length]
        
#         label_poisition = i+sequence_length
#         label = input_data.iloc[label_poisition:label_poisition+offset][target_column]
        
#         sequences.append((sequence, label))
        
#     return sequences


def seed_all(seed = 42):
    # Set the seed for Python's 'random'
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)

    # If using CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Additional settings for reproducibility (if using CUDA)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    

def MSE(y, y_pred):
    return mean_squared_error(y, y_pred)

def RMSE(y, y_pred):
    return math.sqrt(mean_squared_error(y, y_pred))

def MAE(y, y_pred):
    return mean_absolute_error(y, y_pred)

def MAPE(y, y_pred):
    return mean_absolute_percentage_error(y, y_pred)