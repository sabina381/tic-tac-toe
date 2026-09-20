from easydict import EasyDict

import torch
from pathlib import Path
import os

# Environment
STATE_SIZE = (3, 3)
WIN_CONDITION = 3

PATH = Path(os.getcwd()).parent
F_PATH = PATH / "experiments"
F_NAME = "alpz_ttt_01"

ALLOW_TRANSPOSE = True 
DATA_AUGMENTATION = True # 데이터 증강

REWARD_WIN = 1
REWARD_LOSE = -1
REWARD_DRAW = 0.1

POINT_WIN = (REWARD_WIN - REWARD_LOSE)/(REWARD_WIN - REWARD_LOSE)
POINT_LOSE = (REWARD_LOSE - REWARD_LOSE)/(REWARD_WIN - REWARD_LOSE)
POINT_DRAW = (REWARD_DRAW - REWARD_LOSE)/(REWARD_WIN - REWARD_LOSE)

# state
NUM_HISTORY = 0
PLAYER_INFO = True

# enemy agents
AB_DEPTH = 100 # 알파베타 알고리즘
MCS_PO_NUM = 100 # MCS 알고리즘
MCTS_EV_NUM = 100 # MCTS 알고리즘

# net
ACTION_SIZE = STATE_SIZE[0] * STATE_SIZE[1]
STATE_DIM = 2 + NUM_HISTORY * 2 + int(PLAYER_INFO)

CONV_UNITS = 64
RESIDUAL_NUM = 8

# device
if torch.backends.mps.is_built():
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

elif torch.backends.cuda.is_built():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

else:
    DEVICE = torch.device("cpu")

# mcts
C_PUCT = 5
EVAL_CNT = 200
TEMPERATURE = 1.0
TEMPERATURE_DECAY = 0.9

# train
LEARN_RATE = 0.001
GAMMA = 0.1
TOTAL_SP_NUM = 100
SP_NUM_TRAIN = 1  # 셀프 플레이를 수행할 게임 수(오리지널: 25,000)
EXPLORE_REGULATION = 4

# SCHEDULER = optim.lr_scheduler.StepLR(OPTIMIZER, step_size=10, gamma=GAMMA)
CROSS_ENTROPY = torch.nn.CrossEntropyLoss()

BATCHSIZE = 256
TRAIN_EPOCHS = 10
MEM_SIZE = 30000

EPISODES = int(TOTAL_SP_NUM / SP_NUM_TRAIN)

# evaluation
EVAL_NUM_GAME = 20
TEST_NUM_GAME = 10
CRITERIA = 0.5

EVAL_FREQUENCY = 10
PRINT_LOSS_FREQENCY = 10


CONFIG = EasyDict({
    'STATE_SIZE' : STATE_SIZE,
    'WIN_CONDITION' : WIN_CONDITION,
    'PATH' : PATH,
    'F_PATH' : F_PATH,
    'F_NAME' : F_NAME,

    'ALLOW_TRANSPOSE' : ALLOW_TRANSPOSE,
    'DATA_AUGMENTATION' : DATA_AUGMENTATION,

    # state
    'NUM_HISTORY' : NUM_HISTORY,
    'PLAYER_INFO' : PLAYER_INFO,

    # enemy agents
    'AB_DEPTH' : AB_DEPTH,
    'MCS_PO_NUM' : MCS_PO_NUM,
    'MCTS_EV_NUM' : MCTS_EV_NUM,

    # net
    'ACTION_SIZE' : ACTION_SIZE,
    'STATE_DIM' : STATE_DIM,

    'CONV_UNITS' : CONV_UNITS,
    'RESIDUAL_NUM' : RESIDUAL_NUM,

    'DEVICE' : DEVICE,

    # mcts
    'C_PUCT' : C_PUCT,
    'EVAL_CNT' : EVAL_CNT,
    'TEMPERATURE' : TEMPERATURE,
    'TEMPERATURE_DECAY' : TEMPERATURE_DECAY,

    # train
    'LEARN_RATE' : LEARN_RATE,
    'GAMMA' : GAMMA,
    'TOTAL_SP_NUM' : TOTAL_SP_NUM,
    'SP_NUM_TRAIN' : SP_NUM_TRAIN,
    'EXPLORE_REGULATION' : EXPLORE_REGULATION,

    'CROSS_ENTROPY' : CROSS_ENTROPY,

    'BATCHSIZE' : BATCHSIZE,
    'TRAIN_EPOCHS' : TRAIN_EPOCHS,
    'MEM_SIZE' : MEM_SIZE,

    'EPISODES' : EPISODES,

    # evaluation
    'EVAL_NUM_GAME' : EVAL_NUM_GAME,
    'TEST_NUM_GAME' : TEST_NUM_GAME,
    'CRITERIA' : CRITERIA,

    'EVAL_FREQUENCY' : EVAL_FREQUENCY,
    })