# # import
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('/Users/seungyeonlee/Documents/GitHub/24-2-TicTacToe'))))

# from Environment import Environment
# from Environment import State

# # parameter
# env = Environment()


# 1 game play하는 함수
def play_game(player_list):
    is_done = False
    state = State()

    while not is_done:
        player = player_list[0] if state.check_first_player() else player_list[1]

        _, action = player.get_action()
        state, is_done, _ = env.step(state, action)

    # 게임 종료 후 first player 기준 reward, point 계산
    reward = env.first_player_reward(state)

    point = 1 if reward == 1 else 0.5 if reward == 0 else 0
    return point # first player point


# player_list의 두 에이전트의 대국을 통해 첫 번째 알고리즘의 성능 평가
def evaluate_algorithm(label, player_list, num_game):
    total_point = 0
    for i in range(num_game):
        if i % 2 == 0:
            total_point += play_game(player_list)
        else:
            total_point += 1 - play_game(player_list[[1, 0]])

    average_point = total_point / num_game
    print(f"{label}: {average_point}/{total_point}")

    return average_point
