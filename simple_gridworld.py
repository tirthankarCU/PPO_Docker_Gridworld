import numpy as np
import math
'''
0 - move up
1 - move down
2 - move right
3 - move left
'''
ACTION_ARR=[[-1,0],[1,0],[0,1],[0,-1]]
class SimpleGridWorld:
    def __init__(self,size=5,src=[0,1],dest=[4,3],obstacles=[[2,0],[2,1],[2,2],[2,3]],mode='not_human') -> None:
        self.grid=[[0 for i in range(size)] for j in range(size)]
        self.grid[src[0]][src[1]]=1 # 1 is agent
        self.grid[dest[0]][dest[1]]=2 # 2 is destination
        for o in obstacles:
            self.grid[o[0]][o[1]]=3 # 3 is obstacle
        self.curr=src # where the agent is currently.
        self.dest=dest 
        self.obstacles=obstacles
        self.size=size
        self.step=0 
        self.mx_iter=30
        self.obs={'agent':src,'destination':dest,'obstacle':obstacles}
        self.mode=mode 
        self._ip_shape = len(src) + len(dest) + len(obstacles)*2
    def __get_obs(self):
        self.obs['agent']=self.curr
        part1=[self.curr[0],self.curr[1],self.dest[0],self.dest[1]]
        part2=[ o[i] for o in self.obstacles for i in range(2)]
        rectified=np.array(part1+part2)/self.size
        return rectified
    def reset(self,mode_r='not_human'):
        self.__init__(size=5,src=[0,1],dest=[0,3],obstacles=[[2,0],[2,1],[2,2],[2,3]],mode=mode_r)
        return self.__get_obs()
    def printGrid(self):
        for row in self.grid:
            for element in row:
                print(element,end=' ')
            print()
    def action_step(self,action):
        if self.mode=='human':
            self.printGrid()
        self.step+=1
        i_child=self.curr[0]+ACTION_ARR[action][0]
        j_child=self.curr[1]+ACTION_ARR[action][1]
        if self.step>=self.mx_iter:
            return self.__get_obs(),-1,True 
        if i_child==self.obs['destination'][0] and j_child==self.obs['destination'][1]:
            return self.__get_obs(),10,True 
        if i_child<0 or j_child<0 or i_child>=self.size or j_child>=self.size or self.grid[i_child][j_child]==3:
            return self.__get_obs(),-1,False 
        self.grid[self.curr[0]][self.curr[1]]=0
        self.curr[0],self.curr[1]=i_child,j_child
        self.grid[self.curr[0]][self.curr[1]]=1
        return self.__get_obs(),-1,False
    @property
    def ip_shape(self):
        return self._ip_shape 
    @property
    def op_shape(self):
        # 4 actions are possible in this evironment.
        return 4
    @property
    def threshold_reward(self):
        # The ideal reward for this static environment is -7
        return math.inf

if __name__=='__main__':
    env=SimpleGridWorld()
    print(env.ip_shape)
    actions = [1,2,2,2,1,1,1,3]
    for action in actions:
        state,reward,termination = env.action_step(action)
        print(action,reward)
        env.printGrid()