# import
import numpy as np
import copy

from tictactoe_env import Environment
from ResNet import ResidualBlock, Net
from mcts import Mcts


# parameter
SP_GAME_COUNT = 500  # 셀프 플레이를 수행할 게임 수(오리지널: 25,000)
SP_TEMPERATURE = 1.0  # 볼츠만 분포의 온도 파라미터

state_size = (3,3)
reward_dict = {'win':1, 'lose':-1, 'draw':0, 'progress':0}
env = Environment(state_size, reward_dict)

CONV_UNITS = 64
model = Net(state_size, env.num_actions, CONV_UNITS)


# 1번의 게임 play 함수
def play_one_game(model):
    env.reset()
    history = []

    while not env.done:
        state = env.present_state.copy()
        mcts = Mcts(env, model, state, temperature = SP_TEMPERATURE)
        policy, action = mcts.get_action()
        _, reward, _, _ = env.step(action)

        state[[0, 1]] = state[[1, 0]] if not env.player else state[[0, 1]] # player = True 기준, (player_state, enemy_state) 고정
        history.append((state, policy))

    history = [(x[0], x[1], reward) for x in history]

    return history


# self play 함수
def self_play():
    env = copy.deepcopy(env)
    data = []

    for i in range(SP_GAME_COUNT):
        history = play_one_game(model)
        data.extend(history)

    return data