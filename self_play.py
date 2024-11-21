# import
import numpy as np

from tictactoe_env import Environment
from ResNet import ResidualBlock, Net
from mcts import Mcts


# parameter
num_games = 100

state_size = (3,3)
reward_dict = {'win':1, 'lose':-1, 'draw':0, 'progress':0}
env = Environment(state_size, reward_dict)

CONV_UNITS = 64
model = Net(state_size, env.num_actions, CONV_UNITS)

mcts = Mcts(env, model, env.present_state, temperature = 1)


# self play
def self_play(model, num_games):
    data = []

    for i in range(num_games):
        env.reset()
        episode_history = []

        while not env.done:
            state = env.present_state.copy()
            mcts = Mcts(env, model, state, temperature = 1)
            policy, action = mcts.get_action()
            _, reward, _, _ = env.step(action)

            episode_history.append((state, policy))

        result = reward
        
        episode_history = [(x[0], x[1], reward) for x in episode_history]
        data.extend(episode_history)

    return data