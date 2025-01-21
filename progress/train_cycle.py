# import
from ResNet import Net
from self_play import self_play, play_one_game
from train_network import train_network
from evaluate_network import evaluate_network
from evaluate_best_player import evaluate_best_player

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/Users/seungyeonlee/Documents/GitHub/24-2-TicTacToe'))))

from Environment import Environment

# parameter
TRAIN_NUM = 10

STATE_SIZE = (3, 3)
env = Environment()

CONV_UNITS = 64
model = Net(env.num_actions, CONV_UNITS)

###############################
# network train cycle
def train(model):
    for _ in range(TRAIN_NUM):
        self_play(model)
        train_network()
        update_best_player = evaluate_network()

        if update_best_player:
            evaluate_best_player()
