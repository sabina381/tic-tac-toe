# # import
# import torch

# from file_save_load import load_model
# from mcts import Mcts
# from enemy_agents import RandomAgent, AlphaBetaAgent, McsAgent, MctsAgent
# from play_game import evaluate_algorithm

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/Users/seungyeonlee/Documents/GitHub/24-2-TicTacToe'))))

# from Environment import Environment
# from Environment import State

# # parameter
# env = Environment()
# EP_NUM_GAME = 10
# file_name = "model1"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# best player를 평가하는 함수
# 만약 새로운 모델이 best model로 업데이트 된 경우
# 새로운 best model과 다른 알고리즘의 대국을 진행해 평가한다.

def evaluate_best_player():
    model = load_model(f'{file_name}_model_best.pkl').to(device)

    mcts_best = Mcts(model)
    agents_dict = {'random': RandomAgent(),
                    'alpha-beta': AlphaBetaAgent(), 
                    'mcs': McsAgent(),
                    'mcts': MctsAgent()}

    count = 0
    for key in agents_dict.keys:
        agent = agents_dict[key]
        player_list = [mcts_best, agent]
        count += 1
        _ = evaluate_algorithm(f"[{count}/{len(agents_dict)}] vs. {key}", player_list, EP_NUM_GAME)

    
    
        


