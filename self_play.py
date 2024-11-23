# import
import numpy as np

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

# 내부에서 객체 만들어 사용 (한 번 행동을 선택할 때마다 시행해야 함)
mcts = Mcts(env, model, env.present_state, temperature = 1)

# 1번의 게임 play 함수
def play_one_game(model):
    env.reset()
    history = []

    while not env.done:
        state = env.present_state.copy()
        mcts = Mcts(env, model, state, temperature = SP_TEMPERATURE)
        policy, action = mcts.get_action()
        _, reward, _, _ = env.step(action)

        history.append((state, policy))

    result = reward
    history = [(x[0], x[1], reward) for x in history]

    return history


# self play 함수
def self_play():
    data = []

    for i in range(SP_GAME_COUNT):
        history = play_one_game(model)
        data.extend

    return data