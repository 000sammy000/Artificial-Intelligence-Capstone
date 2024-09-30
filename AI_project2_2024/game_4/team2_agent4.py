"""
    teamID=2
    member:
    黃芷柔 110550142
    李宜蓁 110550113
    劉詠 110550039

"""

import STcpClient
import numpy as np
import random
import math

MAP_SIZE = 12


def scan_sheep(player_id, map_state):
    """
    Return a list of [i,j] indicate the distrubution of the player's sheep
    """
    cors = []
    for i in range(12):
        for j in range(12):
            if map_state[i][j] == player_id:
                cors.append([i, j])
    return cors


def dfs(row, col, grid):
    """
    Return:
    the region size of the conected region
    """
    if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] != 1:
        return 0
    grid[row][col] = -1  # Mark cell as visited
    size = 1
    directions = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    for dir in directions:
        size += dfs(row + dir[0], col + dir[0], grid)
    return size


def minimax(cur_player_id, map_state, sheep_state, depth, alpha, beta, max_player_id):
    """
    Return the score the max player can get according to the new board state
    """
    player_list = [1, 2, 3, 4]
    if is_terminal(max_player_id, map_state, sheep_state):
        return -1000

    if depth == 0:
        my_eval=evaluation(max_player_id, map_state, sheep_state)
        return my_eval

    next_player = player_list[cur_player_id % 4]

    if cur_player_id == max_player_id or cur_player_id == player_list[(cur_player_id + 1) % 4]:
        max_score = -1000
        moves = possible_move(next_player, map_state, sheep_state)
        for move in moves:
            new_map, new_sheep = choose_move(next_player, map_state, sheep_state, move)
            score = minimax(next_player, new_map, new_sheep, depth - 1, alpha, beta, max_player_id)
            max_score = max(score, max_score)
            alpha = max(score, alpha)
            if beta <= alpha:
                break
        return max_score

    else:
        min_score = math.inf
        moves = possible_move(next_player, map_state, sheep_state)
        for move in moves:
            new_map, new_sheep = choose_move(next_player, map_state, sheep_state, move)
            score = minimax(next_player, new_map, new_sheep, depth - 1, alpha, beta, max_player_id)
            min_score = min(score, min_score)
            beta = min(score, beta)
            if beta <= alpha:
                break
        return min_score


def count_way(i, j):
    y = [-1, -1, -1, 0, 0, 1, 1, 1]
    x = [-1, 0, 1, -1, 1, -1, 0, 1]
    dir_index = [1, 2, 3, 4, 6, 7, 8, 9]
    way = 0
    for t in range(8):
        if i + x[t] >= 0 and i + x[t] < MAP_SIZE and j + y[t] >= 0 and j + y[t] < MAP_SIZE and mapStat[i + x[t], j + y[t]] == 0:
            way += 1

    return way


def possible_move(cur_player_id, map_state, sheep_state):
    """
    Expand the possible move of the current player
    Return:
        moves: [(j,i), m, d]
    """
    is_player = []
    for i in range(len(map_state)):
        for j in range(len(map_state)):
            if map_state[i][j] == cur_player_id:
                is_player.append((i, j))
    moves = []

    y = [-1, -1, -1, 0, 0, 1, 1, 1]
    x = [-1, 0, 1, -1, 1, -1, 0, 1]
    dir_index = [1, 2, 3, 4, 6, 7, 8, 9]

    for node in is_player:
        i, j = node
        ori_way = count_way(i, j)

        for k in range(8):

            new_i, new_j = i + x[k], j + y[k]
            if (0 <= new_i < MAP_SIZE) and (0 <= new_j < MAP_SIZE) and map_state[new_i][new_j] == 0 and sheep_state[new_i][new_j] == 0:
                while (
                    (0 <= new_i + x[k] < MAP_SIZE)
                    and (0 <= new_j + y[k] < MAP_SIZE)
                    and map_state[new_i + x[k]][new_j + y[k]] == 0
                    and sheep_state[new_i + x[k]][new_j + y[k]] == 0
                ):
                    new_i, new_j = new_i + x[k], new_j + y[k]

                new_way = count_way(new_i, new_j)
                if sheep_state[i][j] > 1:
                    sheeps = sheep_state[i][j] * new_way / (ori_way + new_way)
                    if sheeps < 1:
                        sheeps = 1
                    moves.append([(i, j), int(sheeps), dir_index[k]])

    # print(moves)
    return moves


def choose_move(cur_player_id, map_state, sheep_state, move):
    """
    Update the new board state according to the move
    Return:
        new_map_state, new_sheep_state
    move: [(j,i),m,d]
    """
    d_row = move[0][1]
    d_col = move[0][0]
    dir_i = [-1, 0, 1]
    dir_j = [-1, 0, 1]
    m = move[1]
    d = move[2]
    new_map, new_sheep = map_state, sheep_state
    new_sheep[d_row][d_col] -= m
    new_cor = [d_row, d_col]
    di = dir_i[int((d - 1) / 3)]
    dj = dir_j[(d - 1) % 3]
    while 0 <= new_cor[0] + di < 12 and 0 <= new_cor[1] + dj < 12 and map_state[new_cor[0] + di][new_cor[1] + dj] == 0:
        new_cor = [new_cor[0] + di, new_cor[1] + dj]
    new_map[new_cor[0]][new_cor[1]] = cur_player_id
    new_sheep[new_cor[0]][new_cor[1]] = m
    return new_map, new_sheep


def evaluation(max_player_id, map_state, sheep_state):
    """
    Evaluate the score of the node
    Return:
        score
    """
    # if is_terminal(max_player_id, map_state, sheep_state):
    #     return -1000
    cors = scan_sheep(max_player_id, map_state)
    areas = []
    visited = [[0 for _ in range(12)] for _ in range(12)]
    for cor in cors:
        visited[cor[0]][cor[1]] = 1
    for i in range(12):
        for j in range(12):
            if visited[i][j] == 1:
                size = dfs(i, j, visited)
                areas.append(size)
    sheep_num = []
    for p in cors:
        sheep_num.append(sheep_state[p[0]][p[1]])
    mul = 1
    for i in sheep_num:
        mul *= i
    penalty = 0
    for id in range(1, 4, 1):
        others = []
        if id == max_player_id:
            continue
        cors = scan_sheep(id, map_state)
        visited = [[0 for _ in range(12)] for _ in range(12)]
        for cor in cors:
            visited[cor[0]][cor[1]] = 1
        for i in range(12):
            for j in range(12):
                if visited[i][j] == 1:
                    size = dfs(i, j, visited)
                    others.append(size)
        cur = sum(x for x in others)
        penalty += cur
    score = sum(x**2 for x in areas) - penalty + (mul / 3)
    return score


def is_terminal(max_player_id, map_state, sheep_state):
    """
    Return whether the max player can move or not
    Return:
        True or False
    """
    cors = scan_sheep(max_player_id, map_state)

    if len(cors) == 16:
        return True

    dir_i = [-1, 0, 1]
    dir_j = [-1, 0, 1]
    for cor in cors:
        if sheep_state[cor[0]][cor[1]] == 1:
            continue
        for i in dir_i:
            for j in dir_j:
                if 0 <= cor[0] + i < 12 and 0 <= cor[1] + j < 12 and map_state[cor[0] + i][cor[1] + j] == 0:
                    return False

    return True


def InitPos(mapStat):
    """
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)

    return: init_pos
    init_pos=[x,y],代表起始位置

    """
    dirs = [[-1, 0], [1, 0], [0, 1], [0, -1]]
    boarder = []
    sheep_state = [[0 for _ in range(12)] for _ in range(12)]
    cur_id = 1
    for i in range(12):
        for j in range(12):
            if mapStat[i][j] == 0:
                for dir in dirs:
                    if 0 <= i + dir[0] < 12 and 0 <= j + dir[1] < 12 and mapStat[i + dir[0]][j + dir[1]] == -1 and [i, j] not in boarder:
                        boarder.append([i, j])
            # elif 1<=mapStat[i][j]<=4:
            #     cur_id+=1
            #     sheep_state[i][j]=16
    init_pos = random.choice([boarder[1], boarder[0]])

    return init_pos


def GetStep(playerID, mapStat, sheepStat):
    """
    產出指令

    input:
    playerID: 你在此局遊戲中的角色(1~4)
    mapStat : 棋盤狀態(list of list), 為 12*12矩陣,
            0=可移動區域, -1=障礙, 1~4為玩家1~4佔領區域
    sheepStat : 羊群分布狀態, 範圍在0~16, 為 12*12矩陣

    return Step
    Step : 3 elements, [(x,y), m, dir]
            x, y 表示要進行動作的座標
            m = 要切割成第二群的羊群數量
            dir = 移動方向(1~9),對應方向如下圖所示
            1 2 3
            4 X 6
            7 8 9
    """
    step = [(0, 0), 0, 1]
    """
    Write your code here
    在這裡寫tree
    
    """
    moves = possible_move(playerID, mapStat, sheepStat)
    if len(moves) == 0:
        print("there's no possible moves")
    step = None
    best_score = -math.inf
    depth = 3
    for move in moves:
        new_map, new_sheep = choose_move(playerID, mapStat, sheepStat, move)
        score = minimax(playerID, new_map, new_sheep, depth, -math.inf, math.inf, playerID)
        if score > best_score:
            best_score = score
            step = move
    """print("map")
    print(mapStat)
    print()
    print(f"move = {step}, score = {best_score}, score = {score}")"""
    return step


# player initial
(id_package, playerID, mapStat) = STcpClient.GetMap()
init_pos = InitPos(mapStat)
STcpClient.SendInitPos(id_package, init_pos)

# start game
while True:
    (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
    if end_program:
        STcpClient._StopConnect()
        break
    Step = GetStep(playerID, mapStat, sheepStat)

    STcpClient.SendStep(id_package, Step)
