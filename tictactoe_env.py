# import
from typing import Tuple
import numpy as np

# parameter
state_size = (3,3)
reward_dict = {'win':1, 'lose':-1, 'draw':0, 'progress':0}

# class tictactoe environment
class Environment:
    def __init__(self, state_size:Tuple, reward_dict:dict):
        # env size
        self.state_size = state_size # (3, 3)
        self.n = self.state_size[0] # 3
        self.num_actions = self.n ** 2 # 9

        # state, action
        self.present_state = np.zeros((2, self.n, self.n)) # present_state[0]: state for first player
        self.action_space = np.arange(self.num_actions) # [0, 1, ..., 8] : action idx

        # reward, done
        self.reward_dict = reward_dict
        self.done = False
        
        # 추가
        self.player = True # True: first player


    def step(self, action_idx):
        '''
        action_idx에 따라 게임 진행
        output: next_state, reward, done, is_win
        '''
        x, y = divmod(action_idx, self.n)

        self.present_state[0][x, y] = -1
        next_state = self.present_state

        # 게임 종료 및 승자 확인
        done, is_win = self.is_done(next_state)
        reward = self.check_reward(is_win)

        self.done = done
        self.change_player() # change turn
        
        return next_state ,reward, done, is_win


    def reset(self):
        '''
        game reset
        '''
        self.present_state = np.zeros((2, self.n, self.n))
        self.done = False
        self.player = True

    def render(self, state):
        '''
        print by string
        first player: X / second player: O
        '''
        state = state if self.player else state[[1, 0]]
        state = state.reshape(2, -1)
        check_state_1 = np.array(list(map(lambda x: 'X' if state[0][x] != 0 else '.', self.action_space)))
        check_state_2 = np.array(list(map(lambda x: 'O' if state[1][x] != 0 else '.', self.action_space)))
        combined_array = np.array([check_state_1[i] if check_state_1[i] != '.' else check_state_2[i] for i in self.action_space])

        # string으로 변환하고 game board 형태로 출력
        combined_string = ' '.join(combined_array)
        formatted_string = '\n'.join([combined_string[i:i+6] for i in range(0, len(combined_string), 6)])

        print(formatted_string)
        print("-"*10)
        

    def check_legal_action(self, state):
        '''
        state에서 가능한 action array를 원핫으로 출력
        '''
        state = state[0].reshape(-1)
        legal_actions = np.array(list(map(lambda x: state[x] == 0, self.action_space)))
        return legal_actions


    def is_done(self, state):
        '''
        game의 종료 여부 확인
        is_win: True - win / False - draw
        '''
        is_done, is_win = False, False
        player_state = state[0]

        # 무승부 여부 확인
        if state.sum() == -9:
            is_done = True, False

        # 승리 조건 확인
        axis_diag_sum = np.concatenate([player_state.sum(axis=0), player_state.sum(axis=1), [player_state.trace()], [np.fliplr(player_state).trace()]]) # (8, )
        if -3 in axis_diag_sum:
            is_done, is_win = True, True

        return is_done, is_win


    # 추가 메서드
    def change_player(self):
        '''
        state를 다음 턴으로 돌려줌
        player를 다음 player로 바꿈
        '''
        self.present_state[[0, 1]] = self.present_state[[1, 0]]
        self.player = not self.player


    def check_reward(self, is_win):
        '''
        reward를 주는 함수
        draw, progress: 0 
        first player 기준 reward 제공
        '''
        reward = 0

        if is_win:
            reward = self.reward_dict["win"] if self.player else self.reward_dict["lose"]

        return reward