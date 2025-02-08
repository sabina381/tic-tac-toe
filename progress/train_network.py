# import 
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms

import numpy as np
import pickle
import random

from ResNet import Net
from file_save_load import load_history, save_model, load_model

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/Users/seungyeonlee/Documents/GitHub/24-2-TicTacToe'))))

from Environment import Environment

# parameter
TRAIN_EPOCHS = 100  # 학습 횟수
BATCHSIZE = 64
LEARN_MAX = 0.001

file_name = "model1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_size = (3,3)
env = Environment()

CONV_UNITS = 64
model = Net(state_size, env.num_actions, CONV_UNITS).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_MAX)

'''
차원 수정, history 보고 물려야함
'''
#######################################
# loss function 정의
def loss_function(pred_policy, pred_value, y):
    mse = F.mse_loss(pred_policy, y[0])
    cross_entrophy = -torch.mean(y[1] * torch.log(pred_value))
    return mse + cross_entrophy


# dataset을 만드는 함수
def make_dataset(history):
    batch_size = min(BATCHSIZE, len(history))
    mini_batch = random.sample(history, batch_size)
    states, policies, results = zip(mini_batch)
    X = torch.tensor(states, dtype=torch.float32).to(device) # (batch_size, )
    Y = torch.tensor([policies, results], dtype=torch.float32).to(device) # (batch_size, 2)
    return X, Y


# network train하는 함수
def train_network():
    history = load_history(f'{file_name}_history.pkl')
    model = load_model(f'{file_name}_model_latest.pkl')
    
    for _ in range(TRAIN_EPOCHS):
        X, Y = make_dataset(history)
        pred_policy, pred_value = model.forward(X)
        loss = loss_function(pred_policy, pred_value, Y)
        # 역전파
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

    # 최근 모델 저장
    save_model(f'{file_name}_model_latest.pkl', model)

    # lr,... epoch,... 조절...
        




   
    
