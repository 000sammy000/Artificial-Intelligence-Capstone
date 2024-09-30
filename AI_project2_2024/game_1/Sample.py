import STcpClient
import numpy as np
import random


'''
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=[x,y],代表起始位置
    
'''
def check_boarder(mapStat,i,j):
    cnt=0
    if mapStat[i,j-1]==-1:
        cnt+=1
    if mapStat[i,j+1]==-1:
        cnt+=1
    if mapStat[i-1,j]==-1:
        cnt+=1
    if mapStat[i+1,j]==-1:
        cnt+=1
    
    if cnt>0:
        return 1
    else:
        return 0


def InitPos(mapStat):
    init_pos = [0, 0]
    boarder=[]

    for i in range(12):
        for j in range(12):
            if mapStat[i,j]==0 and check_boarder:
                boarder.append((i,j))
                #print(i,j)
    
    
    random_border = random.choice(boarder)
    i, j = random_border
    random_point = [i, j]
    

    '''
        Write your code here
        找boarder

    '''

    return random_point


def find_next(mapStat,i,j,playerID):
    y=[-1,-1,-1,0,0,1,1,1]
    x=[-1,0,1,-1,1,-1,0,1]
    dir_index=[1,2,3,4,6,7,8,9]
    dir=0
    
    for a in range(8):
        i_new=i
        j_new=j
        find=0
        if i_new+x[a]>=0 and i_new+x[a]<=12 and j_new+y[a]>=0 and j_new+y[a]<=12:
            if mapStat[i_new+x[a],j_new+y[a]]==0:
                find=1

        if find==1:
            dir=dir_index[a]
            break
    
    return dir


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
def GetStep(playerID, mapStat, sheepStat):
    step = [(0, 0), 0, 1]
    '''
    Write your code here
    
    '''
    dir=0
    for i in range(12):
        for j in range(12):
            if playerID==mapStat[i,j] and sheepStat[i,j]!=1:
                dir=find_next(mapStat,i,j,playerID)
                if dir!=0:
                    break
                
        if dir!=0:
            break    
            
    
    step = [(i, j), int(sheepStat[i,j]/2), dir]
    return step


# player initial
(id_package, playerID, mapStat) = STcpClient.GetMap()
init_pos = InitPos(mapStat)
STcpClient.SendInitPos(id_package, init_pos)

# start game
while (True):
    (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
    if end_program:
        STcpClient._StopConnect()
        break
    Step = GetStep(playerID, mapStat, sheepStat)

    STcpClient.SendStep(id_package, Step)
