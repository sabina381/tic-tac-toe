# import
import copy
import numpy as np

from tictactoe_env import Environment

# parameter
state_size = (3,3)
reward_dict = {'win':1, 'lose':-1, 'draw':0, 'progress':0}
env = Environment(state_size, reward_dict)
CPARM = 2

# node
class Node():
    def __init__(self, env, state, policy, value, player:bool):
        self.env = copy.deepcopy(env) # 환경의 method를 사용
        self.legal_actions = self.env.check_available_action(state)
        self.player = player

        # attribute
        self.state = state
        self.childnode_list = []
        self.policy = policy # 사전 확률 (model predict)
        self.visits = value # value 초기값 (model predict)
        self.cum_value = 0

    def select_child(self):
        '''
        PUCT를 이용한 child node 선택 (selection에 해당)
        '''
        # PUTC 공식을 적용하여 값이 낮은 것부터 child node들을 정렬 (마지막이 최댓값을 갖는 노드)
        # 아직 구현 못함 너무 헷갈림
        result = sorted(self.childnode_list, key=lambda c: CPARM * c.p * ))
        return s[-1]


    def add_child(self, action, state):
        '''
        action에 해당하는 child node 추가 (expansion에 해당)
        '''
        self.legal_actions[action] = 0

        env.player = False if self.player else True
        self.env.present_state = state
        next_state = self.env.step(action)

        # policy, value 전달에 zip 활용? 알아봐
        policy, value = self.model.predict(next_state)
        policy = policy * self.legal_actions # 가능한 action에 대한 정책으로 전달

        new_node = Node(self.env, copy.deepcopy(state), policy, value, player = not self.player)
        self.childnode_list.append(new_node)

        return new_node


    def update(self, result):
        '''
        이 노드의 값 업데이트 (backpropagation에 해당)
        '''
        self.visits += 1
        self.cum_value += result