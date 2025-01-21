# # import
# import numpy as np
# import random
# import math

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/Users/seungyeonlee/Documents/GitHub/24-2-TicTacToe'))))

# from Environment import Environment

# # parameters
# env = Environment()

# AB_DEPTH = 100 # 알파베타 알고리즘

# MCS_PO_NUM = 30 # MCS 알고리즘

# MCTS_EV_NUM = 100 # MCTS 알고리즘

# random agent ################################
class RandomAgent:
    __slots__ = ()

    def get_action(self, state):
        return state.get_ramdom_action()


# alpha-beta(minimax) agent ################################
class AlphaBetaAgent:
    __slots__ = ('player', 'best_action')

    def __init__(self, root_state):
        self.player = root_state.check_first_player()
        self.best_action = None
        

    def get_action(self, state):
        self.minimax(state, AB_DEPTH, -np.Inf, np.Inf)
        return self.best_action


    def minimax(self, state, depth, alpha, beta):
        is_done, _ = state.check_done()
        reward = 0

        legal_actions = np.where(state.get_legal_actions()==1)

        if is_done or (depth == 0):
            reward = env.get_reward(state)
            return reward

        if state.check_first_player() == self.player: # max player
            max_eval = -np.Inf

            for action in legal_actions:
                next_state, _, _ = env.step(state)

                eval = self.minimax(next_state, depth-1, alpha, beta)
                
                if eval > max_eval:
                    best_action = action
                    max_eval = eval
                
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

                if depth == AB_DEPTH: # 최상위 호출에서만 best action 저장
                    self.best_action = best_action
            
            return max_eval

        else: # min player
            min_eval = np.Inf
            for action in legal_actions:
                next_state, _, _ = env.step(state)

                eval = self.minimax(next_state, depth-1, alpha, beta)
                min_eval = min(min_eval, eval)

                beta = min(beta, eval)
                if beta <= alpha:
                    break
            
            return min_eval


# MCS agent ################################
class McsAgent():
    __slots__ = ()

    def get_action(self, state):
        legal_actions = np.where(state.get_legal_actions() == 1)
        value_list = np.zeros(len(legal_actions))

        for i, action in enumerate(legal_actions):
            next_state = env.step(state, action)

            for _ in range(MCS_PO_NUM):
                value_list[i] += - self.playout(next_state)

        max_idx = self.get_max_idx(value_list)
        return legal_actions[max_idx]


    def playout(self, state):
        is_done, _ = state.check_done()

        if is_done:
            return env.get_reward(state)

        action = state.get_random_action()
        next_state, _, _ = env.step(state, action)

        return - self.playout(next_state)


    def get_max_idx(self, value_list):
        return value_list.index(max(value_list))


# MCTS Agent ################################
class MctsAgent():
    __slots__ = ('Node')

    def __init__(self):
        # define Node class ##################
        class Node:
            __slots__ = ('state', 'n', 'w', 'child_nodes')
            
            def __init__(self, state):
                self.state = state 
                self.n = 0 # visit count
                self.w = 0 # cumulative sum of values
                self.child_nodes = None

            def evaluate(self):
                is_done, _ = self.state.check_done()

                # 게임 종료 시 승패 여부에 따라 value 업데이트
                if is_done:
                    value = env.get_reward(self.state)
                    self.w += value
                    self.n += 1
                    return value

                # child node가 없는 경우 확장
                if not self.child_nodes:
                    value = self.playout(self.state)

                    self.w += value
                    self.n += 1

                    # expand child node
                    if self.n == 10:
                        self.expand()

                    return value

                # end node가 아니고, child node가 있는 경우 -> 전개
                else:
                    next_child_node = self.get_next_child_node()
                    value = - next_child_node.evaluate()

                    self.w += value
                    self.n += 1

                    return value


            def expand(self):
                '''
                Expand child node
                '''
                legal_actions = np.where(self.state.get_legal_actions() == 1)
                self.child_nodes = []

                for action in legal_actions:
                    next_state, _, _ = env.step(self.state, action)
                    self.child_nodes.append(Node(next_state))


            def get_next_child_node(self):
                '''
                UCB1이 가장 큰 child node를 선택
                '''
                node_scores = np.array(list(map(lambda c: c.n, self.child_nodes)))

                # 방문 횟수가 0인 child node 반환
                if np.any(node_scores == 0):
                    zero_idx = random.choice(np.where(node_scores == 0))
                    return self.child_nodes[zero_idx]

                total_scores = np.sum(node_scores)
                
                # UCB1 계산 함수
                def ucb1(c):
                    return -c.w/c.n + (2*math.log(total_scores)/c.n)**0.5

                # ucb1 값에 따라 정렬한 child nodes list (마지막이 최댓값을 갖는 child node)
                ucb1_sorted = sorted(self.child_nodes, key = lambda c: ucb1(c))

                return ucb1_sorted[-1]


            def playout(self, state):
                is_done, _ = state.check_done()

                if is_done:
                    return env.get_reward(state)

                action = state.get_random_action()
                next_state, _, _ = env.step(state, action)

                return - self.playout(next_state)
        #########################################

        self.Node = Node()


    def get_action(self, state):
        root_node = self.Node(state)
        root_node.expand()

        for _ in range(MCTS_EV_NUM):
            root_node.evaluate()

        legal_actions = np.where(state.get_legal_actions() == 1)

        child_node_scores = list(map(lambda c: c.n, root_node.child_nodes))
        max_idx = self.get_max_idx(child_node_scores)
        return legal_actions[max_idx]


    def get_max_idx(self, value_list):
        return value_list.index(max(value_list))

    