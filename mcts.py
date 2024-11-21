# import
import copy
import numpy as np

from tictactoe_env import Environment
from ResNet import ResidualBlock, Net

# parameter
C_PUCT = 1.0
EVAL_CNT = 50

state_size = (3,3)
reward_dict = {'win':1, 'lose':-1, 'draw':0, 'progress':0}
env = Environment(state_size, reward_dict)


# define Mcts class
class Mcts:
    def __init__(self, env, model, state, temperature):
        self.env = copy.deepcopy(env)
        self.model = model
        self.state = state
        self.temperature = temperature
        self.legal_actions = self.env.check_legal_action(self.state)

        # define Node class 
        class Node:
            def __init__(self, mcts, state, p):
                self.mcts = mcts # Mcts 객체 참조 저장

                self.state = state
                self.p = p # policy
                self.n = 0 # count
                self.w = 0 # cumulate value
                self.child_nodes = None

            def evaluate(self):
                is_done, is_win = self.mcts.env.is_done(self.state)

                # 게임 종료 시 승패에 따라 value 계산
                if is_done:
                    value = 1 if is_win else 0

                    self.w += value
                    self.n += 1
                    return value

                # child node가 없는 경우 => 확장
                elif not self.child_nodes:
                    # model을 통해 policy와 value 얻음
                    policies, value = self.predict(model, self.state)

                    self.w += value
                    self.n += 1

                    # expand child node
                    self.child_nodes = []
                    legal_actions = self.mcts.legal_actions

                    for action, policy in zip(legal_actions, policies):
                        next_state, _, _, _ = self.mcts.env.step(action)
                        self.child_nodes.append(Node(self.mcts, next_state, policy))

                    return value

                # end node가 아니고, child node가 있는 경우 => 전개
                else:
                    value = -self.next_child_node().evaluate()

                    self.w += value
                    self.n += 1
                    return value
            

            def next_child_node(self):
                '''
                PUCB에 따라 child node를 선택
                '''
                node_scores = list(map(lambda c: c.n, self.child_nodes))
                scores = sum(node_scores)
                # pucb 값에 따라 정렬한 child nodes list (마지막이 최댓값을 갖는 child node)
                pucb_sorted = sorted(self.child_nodes, key = lambda c: (-c.x / c.n if c.n else 0.0) + C_PUCT * c.p * sqrt(scores) / (1 + c.n))

                return pucb_sorted[-1]


            def predict(self):
                '''
                model을 통해 policy와 value 계산
                '''
                x = self.state # 차원 맞추기
                # x = x.unsqueeze(0) 아마도...
                policies, value = model.forward(x)
                policies = policies * self.mcts.legal_actions # legal action에 대한 policy
                policies /= sum(policies) if sum(policies) else 1 # 합계 1의 확률분포로 변환

                return policies, value

        self.Node = Node # Node 객체 생성

    ########################
    # methods of Mcts
    def get_policy(self):
        '''
        MCTS에 따라 policy 계산
        '''
        root_node = self.Node(self, self.state, 0) # Mcts 객체 self 전달

        for _ in range(EVAL_CNT):
            root_node.evaluate()

        scores = [c.n for c in root_node.child_nodes]
        if self.temperature == 0: # 최대값인 경우에만 1로 지정
            action = np.argmax(scores)
            scores = np.zeros(len(scores))
            scores[action] = 1

        else: # 볼츠만 분포를 기반으로 분산 추가
            scores = self.boltzman(scores, self.temperature)

        return scores

 
    def boltzman(self, xs, temperature):
        '''
        볼츠만 분포
        '''
        xs = [x ** (1/temperature) for x in xs]
        return [x/sum(xs) for x in xs]


    def get_action(self):
        '''
        MCTS를 통해 얻은 policy에 따른 action 선택
        '''
        policy = self.get_policy()
        action = np.random.choice(self.legal_actions, p=policy)
        return policy, action
