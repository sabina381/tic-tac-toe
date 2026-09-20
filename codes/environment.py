# import
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import os

# parameter
from config import *
from state import State

#class
class Environment:
    __slots__ = ('state_size', 'win_condition', 'n', 'num_actions', 'action_space', 'reward_dict', 'state')

    def __init__(self, state_size:Tuple, win_condition:int):
        self.state_size = state_size
        self.win_condition = win_condition
        self.n = self.state_size[0]
        self.num_actions = self.n ** 2
        self.action_space = np.arange(self.num_actions)

        self.reward_dict = {'win': REWARD_WIN, 'lose': REWARD_LOSE, 'draw': REWARD_DRAW}

        self.state = State

    def step(self, present_state, action_idx :int):
        '''
        present_state에 대해 action_idx의 행동에 따라 게임을 한 턴 진행시키고
        next_state, is_done, is_lose를 반환한다.
        '''
        next_state = present_state.next(action_idx)
        is_done, is_lose = next_state.check_done()

        return next_state, is_done, is_lose


    def get_reward(self, final_state):
        '''
        게임이 종료된 state에 대해 last player의 reward를 반환한다.
        final_state: 게임이 종료된 state
        '''
        _, is_lose = final_state.check_done()
        reward = self.reward_dict['lose'] if is_lose else self.reward_dict['draw']

        return reward


    def get_first_reward(self, final_state):
        '''
        게임이 종료된 state에 대해 first player의 reward를 반환한다.
        final_state: 게임이 종료된 state
        note: final_state가 is_lose라면, 해당 state에서 행동할 차례였던 플레이어가 패배한 것.
        '''
        is_first_player = final_state.check_first_player()
        _, is_lose = final_state.check_done()
        if is_lose:
            reward = self.reward_dict['lose'] if is_first_player else -self.reward_dict['lose']
        else:
            reward = self.reward_dict['draw']

        return reward


    def render(self, state):
        '''
        입력받은 state를 문자열로 출력한다.
        ○: first_player, ●: second_player
        '''
        is_first_player = state.check_first_player()
        board = state.my_state - state.enemy_state if is_first_player else state.enemy_state - state.my_state
        board = board.reshape(-1)
        board_list = list(map(lambda x: '○' if board[x] == 1 else '●' if board[x] == -1 else '.', self.action_space))

        board_string = ' '.join(board_list)
        formatted_string = '\n'.join([board_string[i:i+(self.n * 2)] for i in range(0, len(board_string), self.n * 2)])

        print(formatted_string)

    ######## play game method ######
    def play_one_game(self, player_list:list, is_show=False, is_save=False):
        is_done = False
        state = self.state()

        data = []
        n_steps = 0

        while not is_done:
            n_steps += 1
            player = player_list[0] if state.check_first_player() else player_list[1]
            
            if 'model' in player.__slots__:
                policy= player.get_policy(state)

                legal_actions = state.get_legal_actions()
                policies = [0] * self.num_actions
                for act, p in zip(legal_actions, policy):
                    policies[act] = p
                    
                if is_save:
                    data.append((state, policies))

                max_actions = np.where(policies == np.max(policies))[0]
                action = np.random.choice(max_actions)
                
                state, is_done, _ = self.step(state, action)

            else:
                action = player.get_action(state)
                state, is_done, _ = self.step(state, action)

            if is_show:
                self.render(state)
                print("-"*30)

        # 게임 종료 후 first player 기준 reward, point 계산
        reward = self.get_first_reward(state)
        point = POINT_WIN if reward == REWARD_WIN else POINT_DRAW if reward == REWARD_DRAW else POINT_LOSE
        if is_show:
            print(f"first reward: {reward}")

        if is_save:
            return point, data
        else:
            return point, None # first player point


    def evaluate_first_algorithm(self, label:str, player_list:list, num_game:int):
        inverse_player_list = [None, None]
        inverse_player_list[0], inverse_player_list[1] = player_list[1], player_list[0]
        total_point = 0
        cnt_win = 0
        cnt_draw = 0

        print_frequency = num_game / 5

        for i in range(num_game):
            if i % 2 == 0:
                print("- black: latest -")
                point, _ = self.play_one_game(player_list, is_show=True)
                total_point += point
                cnt_win += 1 if point == POINT_WIN else 0

            else:
                print("- black: best -")
                point, _ = self.play_one_game(inverse_player_list, is_show=True)
                total_point += 1 - point
                cnt_win += 1 if (-point) == POINT_WIN else 0

            cnt_draw += 1 if point == POINT_DRAW else 0

            # change player
            player_list[0].player = not player_list[0].player
            player_list[1].player = not player_list[1].player

            if (i+1) % print_frequency == 0:
                print(f"Evaluate {label}: {num_game}/{i+1}")

        average_point = total_point / num_game
        print(f"{label}: {round(average_point, 5)} / win:{cnt_win} / draw:{cnt_draw} / lose:{num_game-cnt_win-cnt_draw}")

        return average_point, cnt_win, cnt_draw


    def save_game_image(self, file_name, data):
        fig, axes = plt.subplots(len(data), 2, figsize=(12, 6*len(data)))
        for num in range(len(data)):
            
            # 히트맵 그리기
            policy = np.array(data[num][1]).reshape(self.n, self.n)
            sns.heatmap(policy, annot=True, fmt='.2f', cmap='YlGnBu', cbar=True, ax=axes[num, 1])
            axes[num, 1].set_title('Mcts Policy')

            # 격자 그리기
            first_player = data[num][0].check_first_player()
            board = data[num][0].get_total_state() if first_player else -data[num][0].get_total_state()

            axes[num, 0].set_facecolor('skyblue')
            axes[num, 0].set_title(f"Mcts is first: {first_player}")
            for x in range(self.n):
                axes[num, 0].plot([x, x], [0, self.n-1], color="black", linewidth=1, zorder = 1)  # 세로선
                axes[num, 0].plot([0, self.n-1], [x, x], color="black", linewidth=1, zorder = 1)  # 가로선

            for i in range(self.n):
                for j in range(self.n):
                    if board[i, j] == 1:  # 흑돌
                        circle = plt.Circle((j, self.n-1 - i), 0.4, color="black", zorder = 2)
                        axes[num, 0].add_artist(circle)
                    elif board[i, j] == -1:  # 백돌
                        circle = plt.Circle((j, self.n-1 - i), 0.4, facecolor='white', edgecolor="black", linewidth=1.5, zorder = 2)
                        axes[num, 0].add_artist(circle)

            axes[num, 0].set_xlim(-0.5, self.n - 0.5)
            axes[num, 0].set_ylim(-0.5, self.n - 0.5)
            axes[num, 0].set_aspect('equal')
            
        # 레이아웃 조정 및 저장
        # 저장할 디렉토리 경로 생성
        output_dir = os.path.join(F_PATH, "game_images")
        os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 없으면 생성

        # 이미지 파일 경로 설정
        output_file = os.path.join(output_dir, f"{file_name}.png")

        # 그래프 저장
        plt.tight_layout()
        plt.savefig(output_file)  # 이미지 파일로 저장
        plt.close()               # 메모리 해제

        print(f"<< Game image succefully saved in \'game_images/{file_name}.png\'. >>")
