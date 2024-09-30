#三個對手寫在同一個level
import STcpClient
import numpy as np
import random

'''
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=[x,y],代表起始位置
    
'''
MAP_SIZE=15

def InitPos(mapStat):
    init_pos = [0, 0]
    '''
        Write your code here

    '''
    border_cells = []
    for  i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
                # left                  # right                 #up                         #down
            if ((i-1)>= 0 and mapStat[i-1][j] == -1) or ((i+1) <MAP_SIZE and mapStat[i+1][j] == -1) or ((j-1)>=0 and mapStat[i][j-1] == -1) or ((j+1) < MAP_SIZE and mapStat[i][j+1]):
                if mapStat[i][j] == 0:
                    border_cells.append((i, j))
    if border_cells:
        init_pos = random.choice(border_cells)
    
    return init_pos


'''
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
'''
def count_way(i,j):
    y=[-1,-1,-1,0,0,1,1,1]
    x=[-1,0,1,-1,1,-1,0,1]
    dir_index=[1,2,3,4,6,7,8,9]
    way=0
    for t in range(8):
        if  i+x[t]>=0 and i+x[t]<MAP_SIZE and j+y[t]>=0 and j+y[t]<MAP_SIZE and mapStat[i+x[t],j+y[t]]==0:
            way+=1

    return way

def poss_move(playerID, mapStat, sheepStat):
    is_player = []
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            if mapStat[i][j] == playerID:
                is_player.append((i,j))
    moves = []
    
    y=[-1,-1,-1,0,0,1,1,1]
    x=[-1,0,1,-1,1,-1,0,1]
    dir_index=[1,2,3,4,6,7,8,9]
        
    for node in is_player:
        i, j = node
        ori_way=count_way(i,j)
        
        for k in range(8):

            new_i, new_j = i + x[k], j + y[k]
            if (0 <= new_i < MAP_SIZE) and (0 <= new_j < MAP_SIZE) and mapStat[new_i][new_j] == 0:
                while (0 <= new_i+x[k] < MAP_SIZE) and (0 <= new_j+y[k] < MAP_SIZE) and mapStat[new_i+x[k]][new_j+y[k]] == 0:
                    new_i, new_j = new_i + x[k], new_j + y[k]

                new_way=count_way(new_i,new_j)
                if sheepStat[i][j] >1:
                    sheeps=0
                    if new_way+ori_way-1>new_way:
                        sheeps=sheepStat[i][j]*new_way/(ori_way-1+new_way)
                        if sheeps<1:
                            sheeps=1
                    else:
                        sheeps=sheepStat[i][j]-1
                    moves.append([(i, j), int(sheeps), dir_index[k]])
                
                
            

    #print(moves)
    return moves

def gameover(move,sheepStat):
    over = True
    i,j= move[0]
    if sheepStat[i][j] != 1:
        if ((i-1) >= 0 and (j-1) >= 0) and sheepStat[i-1][j-1] == 0:#還沒有有人放羊
            over = False
        elif ((j-1) >= 0 ) and sheepStat[i][j-1] == 0:
            over = False
        elif ((i+1) <MAP_SIZE and (j-1) >= 0) and sheepStat[i+1][j-1] == 0:
            over = False
        elif  (i-1) >= 0 and sheepStat[i-1][j] == 0:
            over = False
        elif  (i+1) < MAP_SIZE and sheepStat[i+1][j] == 0:
            over = False
        elif ((j+1) <MAP_SIZE and (i-1) >= 0) and sheepStat[i-1][j+1] == 0:
            over = False
        elif ((j+1) < MAP_SIZE ) and sheepStat[i][j+1] == 0:
            over = False
        elif ((i+1) <MAP_SIZE and (j+1) < MAP_SIZE) and sheepStat[i+1][j+1] == 0:
            over = False
    return over #在(i,j)這個點還有動作可以走嗎


def dfs(i, j,mp,playerID,visit):
    if i < 0 or i >= len(mp) or j < 0 or j >= len(mp[0]) or mp[i][j] != playerID or visit[i][j]!=0:
        return 0
    
    # 将当前格子标记为已访问
    visit[i][j] = 1
    
    # 计算当前陆地的面积
    area = 1
    
    area += dfs(i+1, j, mp, playerID,visit)
    area += dfs(i-1, j, mp, playerID,visit)
    area += dfs(i, j+1, mp, playerID,visit)
    area += dfs(i, j-1, mp, playerID,visit)
    
    return area
def evaluation(playerID,mapStat):
    visit = np.zeros((MAP_SIZE,MAP_SIZE))
    total = 0
    for i in range(MAP_SIZE):
        for j in range(MAP_SIZE):
            if mapStat[i][j] == playerID:
                total += dfs(i,j,mapStat,playerID,visit)
    return total

def update(mapStat,sheepStat,move,playerID):
    pos, num_sheep, dir = move
    i, j = pos

    # 定义方向增量
    dir_delta = {
        1: (-1, -1),
        2: (0, -1),
        3: (1, -1),
        4: (-1, 0),
        6: (1, 0),
        7: (-1, 1),
        8: (0, 1),
        9: (1, 1)
    }

    # 循环更新位置直到遇到障碍或到达边界
    delta_i, delta_j = dir_delta[dir]
    while 0 <= (i+delta_i) < MAP_SIZE and 0 <= (j+delta_j) < MAP_SIZE:
        i += delta_i
        j += delta_j
        if mapStat[i][j] != 0:
            mapStat[i-delta_i][j-delta_j] = playerID
            sheepStat[i-delta_i][j-delta_j] = num_sheep
        

    return mapStat, sheepStat
    
def minimax(playerID,mapStat,sheepStat,move, depth, maximizing_player, alpha, beta):
    #需要將move加到mapStat跟sheepStat嗎?要 poss也要改成child node之後的動作
    
    if depth == 0 or gameover(move,sheepStat):
        if depth ==0:
            mapStat,sheepStat = update(mapStat,sheepStat,move,playerID)
        my_eval=evaluation(playerID,mapStat)
        largest_other=0
        for t in range(3):
            other_eval=evaluation((playerID+t)%4+1,mapStat)
            if largest_other<other_eval:
                largest_other=other_eval
            
        return my_eval
    
    # 執行動作的那個點還可以執行(not gameover state)
    #mapStat,sheepStat = update(mapStat,sheepStat,move,playerID)
    
    #print(move)
    if maximizing_player:
        max_eval = float('-inf')
        total_eval=0
        moves = poss_move(playerID,mapStat,sheepStat)
        for move in moves:
            eval = minimax(playerID,mapStat,sheepStat,move, depth - 1, False, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(eval, alpha)
            if beta <= alpha:
                break

        return max_eval
    else:
        mapStat,sheepStat = update(mapStat,sheepStat,move,playerID)

        min_eval = float('inf')
        moves = poss_move(playerID%4+1,mapStat,sheepStat)
        for move in moves:
            mapStat,sheepStat = update(mapStat,sheepStat,move,playerID%4+1)
            eval = minimax(playerID,mapStat,sheepStat,move, depth - 1,True ,alpha , beta)
            i,j=move[0]
            mapStat[i,j]=0
            sheepStat[i,j]=0
            min_eval = min(min_eval, eval)
            beta = min(eval, beta)
            if beta <= alpha:
                break
        moves = poss_move((playerID+1)%4+1,mapStat,sheepStat)
        for move in moves:
            mapStat,sheepStat = update(mapStat,sheepStat,move,(playerID+1)%4+1)
            eval = minimax(playerID,mapStat,sheepStat,move, depth - 1,True ,alpha , beta)
            i,j=move[0]
            mapStat[i,j]=0
            sheepStat[i,j]=0
            min_eval = min(min_eval, eval)
            beta = min(eval, beta)
            if beta <= alpha:
                break
        moves = poss_move((playerID+2)%4+1,mapStat,sheepStat)
        for move in moves:
            mapStat,sheepStat = update(mapStat,sheepStat,move,(playerID+2)%4+1)
            eval = minimax(playerID,mapStat,sheepStat,move, depth - 1,True ,alpha , beta)
            i,j=move[0]
            mapStat[i,j]=0
            sheepStat[i,j]=0
            min_eval = min(min_eval, eval)
            beta = min(eval, beta)
            if beta <= alpha:
                break
        
        return min_eval           

def GetStep(playerID, mapStat, sheepStat):
    #step = [(0, 0), 0, 1]
    '''
    Write your code here
    
    '''
    #choose a position of player than get the movement of that position?
    best_move = [(0, 0), 0, 1]
    max_score = float('-inf')
    moves = poss_move(playerID,mapStat,sheepStat)
    """depth=3
    if round>5:
        depth=1"""
    for move in moves: #只有初始點的moves
        i,j=move[0]
        depth=3
        if round>=5:
            if sheepStat[i,j]<4:
                depth=1
        score = minimax(playerID,mapStat,sheepStat,move, depth, maximizing_player=False, alpha=float('-inf'), beta=float('inf'))
        if score > max_score:
            max_score = score
            best_move = move
    if best_move == [(0, 0), 0, 1]:
        best_move = moves[0]
    return best_move
    
    #return step


# player initial
(id_package, playerID, mapStat) = STcpClient.GetMap()
print(len(mapStat),len(mapStat[0]))
init_pos = InitPos(mapStat)
STcpClient.SendInitPos(id_package, init_pos)
round=0

# start game
while (True):
    round+=1
    print("round:",round)
    (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
    if end_program:
        STcpClient._StopConnect()
        break
    Step = GetStep(playerID, mapStat, sheepStat)

    STcpClient.SendStep(id_package, Step)

