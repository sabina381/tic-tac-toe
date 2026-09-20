import numpy as np
import copy


from config import *
from utils import env

class State():
    __slots__ = ('env', 'my_state', 'enemy_state', 'previous_action', 'previous_state', 'history')

    def __init__(self, my_state=None, enemy_state=None, previous_action=None, previous_state=None):

        self.my_state = my_state if my_state != None else [0] * (env.n ** 2)
        self.enemy_state = enemy_state if enemy_state != None else [0] * (env.n ** 2)
        self.previous_action = previous_action
        self.previous_state = previous_state

        self.my_state = np.array(self.my_state).reshape(env.state_size)
        self.enemy_state = np.array(self.enemy_state).reshape(env.state_size)
        self.history = self.make_history(NUM_HISTORY)


    def make_history(self, num_history):
        history = [self.my_state, self.enemy_state]
        temp = self.previous_state
        
        if num_history > 0:
            for _ in range(num_history):
                if temp is not None:
                    history.extend([temp.my_state, temp.enemy_state])
                    temp = temp.previous_state
                else:
                    history.extend([np.zeros(env.state_size), np.zeros(env.state_size)])

        return history

    def total_pieces_count(self):
        '''
        이 state의 전체 돌의 개수를 반환한다.
        '''
        total_state = self.my_state + self.enemy_state
        return np.sum(total_state)


    def get_legal_actions(self):
        '''
        이 state에서 가능한 action idx의 리스트를 반환한다.
        '''
        total_state = (self.my_state + self.enemy_state).reshape(-1)
        legal_actions = np.where(total_state == 0)[0]
        return legal_actions


    def get_onehot_legal_actions(self):
        '''
        이 state에서 가능한 action을
        one-hot encoding 형식의 array로 반환한다.
        '''
        legal_actions = self.get_legal_actions()
        onehot_legal_actions = np.array([legal_actions[x] == 0 for x in env.action_space], dtype = int)
        return onehot_legal_actions


    def check_line_done(self, line):
        k = WIN_CONDITION
        line = np.pad(line, (0, 1), 'constant', constant_values=0)

        window = np.sum(line[:k-1])
        for i in range(k-1, len(line)):
            window += line[i] - line[i - k]
            if window == k:
                # 육목 확인 조건
                window += line[i+1] - line[i+1 - k]
                if window == k:
                    break
                else:
                    return True

        return False


    def check_lose(self):
        '''
        이 state의 lose 여부를 반환한다.
        note: 상대가 행동한 후, 자신의 행동을 하기 전 이 state를 확인한다.
        따라서 이전 state에서 상대의 행동으로 상대가 이긴 경우는 이 state의 플레이어가 진 경우이다.
        '''
        if self.total_pieces_count() < WIN_CONDITION * 2 - 1:
            return False

        is_lose = False
        x, y = np.divmod(self.previous_action, env.n)

        # check row
        row = self.enemy_state[x, :]
        if not np.sum(row) < WIN_CONDITION:
            is_lose = self.check_line_done(row)
            # print(f"row: {is_lose}")
            if is_lose:
                return is_lose

        # check col
        col = self.enemy_state[:, y]
        if not np.sum(col) < WIN_CONDITION:
            is_lose = self.check_line_done(col)
            # print(f"col: {is_lose}")
            if is_lose:
                return is_lose

        # check diag
        diag = np.diag(self.enemy_state, k = y - x)
        if not np.sum(diag) < WIN_CONDITION:
            is_lose = self.check_line_done(diag)
            # print(f"diag: {is_lose}")
            if is_lose:
                return is_lose

        # check anti diag
        anti_diag = np.diag(np.fliplr(self.enemy_state), k = env.n - 1 - y - x)
        if not np.sum(anti_diag) < WIN_CONDITION:
            is_lose = self.check_line_done(anti_diag)
            # print(f"anti diag: {is_lose}")
            if is_lose:
                return is_lose

        return is_lose


    def check_draw(self):
        if self.total_pieces_count() == self.env.num_actions:
            return True

        return False


    def check_done(self):
        is_lose = self.check_lose()
        is_done = is_lose or self.check_draw()

        return is_done, is_lose


    def next(self, action_idx):
        '''
        주어진 action에 따라 다음 state를 생성한다.
        note: 다음 state는 상대의 차례이므로 state 순서를 바꾼다.
        '''
        x, y =np.divmod(action_idx, self.env.n)
        state = self.my_state.copy()
        state[x, y] = 1

        state = list(state.reshape(-1))
        enemy_state = list(copy.copy(self.enemy_state).reshape(-1))

        return State(enemy_state, state, action_idx, self)


    def check_first_player(self):
        '''
        이 state의 플레이어가 first player인지 확인한다.
        '''
        return (self.total_pieces_count() % 2) == 0


    def get_random_action(self):
        '''
        이 state에서 가능한 action 중 랜덤으로 action을 반환한다.
        '''
        legal_actions = self.get_legal_actions()
        action = np.random.choice(legal_actions)
        return action


    def get_total_state(self):
        '''
        history에 넣을 전체 게임보드 state
        자신의 수: 1 / 상대의 수: -1 / 빈칸: 0
        '''
        # return (self.state, self.enemy_state)
        return self.my_state - self.enemy_state


    def render(self):
        '''
        이 state를 렌더링한다.
        '''
        env.render(self)