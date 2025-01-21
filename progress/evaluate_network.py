# # import
# import torch

# from mcts import Mcts
# from file_save_load import load_model, save_model
# from play_game import evaluate_algorithm

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/Users/seungyeonlee/Documents/GitHub/24-2-TicTacToe'))))

# from Environment import Environment

# # parameter
# NUM_GAME = 10
# TEMPERATURE = 1.0 # 볼츠만 분포

# file_name = "model1"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# STATE_SIZE = (3,3)
# env = Environment()

# CONV_UNITS = 64

##############################
# network 평가하는 함수
def evaluate_network():
    model_latest = load_model(f'{file_name}_model_latest.pkl').to(device)
    model_best = load_model(f'{file_name}_model_best.pkl').to(device)

    mcts_latest = Mcts(model_latest, TEMPERATURE)
    mcts_best = Mcts(model_best, TEMPERATURE)

    player_list = [mcts_latest, mcts_best]

    # 대전
    average_point = evaluate_algorithm("Evaluate network", player_list, NUM_GAME)

    # best player 교체
    if average_point > 0.5:
        save_model(f'{file_name}_model_best.pkl', model_latest)
        return True
    else:
        return False




