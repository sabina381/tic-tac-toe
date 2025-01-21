import numpy as np
from State import State
# parameter
STATE_SIZE = (3, 3)

# Environment class
class Environment:
    __slots__ = ('n', 'num_actions', 'action_space', 'reward_dict')
    
    def __init__(self):
        self.n = STATE_SIZE[0]
        self.num_actions = self.n ** 2
        self.action_space = np.arange(self.num_actions)
        self.reward_dict = {'win': 1, 'lose': -1, 'draw': 0}


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


    def first_player_reward(self, final_state):
        '''
        게임이 종료된 state에 대해 first player의 reward를 반환한다.
        final_state: 게임이 종료된 state
        note: final_state가 is_lose라면, 해당 state에서 행동할 차례였던 플레이어가 패배한 것.
        '''
        is_first_player = final_state.check_first_player()
        reward = self.get_reward(final_state) if is_first_player else -self.get_reward(final_state)
        
        return reward


    def render(self, state):
        '''
        입력받은 state를 문자열로 출력한다.
        X: first_player, O: second_player
        '''
        is_first_player = state.check_first_player()
        board = state.state - state.enemy_state if is_first_player else state.enemy_state - state.state
        board = board.reshape(-1)
        board_list = list(map(lambda x: 'X' if board[x] == 1 else 'O' if board[x] == -1 else '.', self.action_space))

        board_string = ' '.join(board_list)
        formatted_string = '\n'.join([board_string[i:i+6] for i in range(0, len(board_string), 6)])

        print(formatted_string)





