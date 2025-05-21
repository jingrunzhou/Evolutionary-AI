"""
迷宫导航挑战环境
用于测试AI的空间认知和路径规划能力
"""
import numpy as np
from envs.base import BaseEnvironment

class MazeEnvironment(BaseEnvironment):
    def __init__(self, size=10):
        """
        初始化迷宫环境
        :param size: 迷宫大小(size x size)
        """
        super().__init__()
        self.size = size
        self.state_size = size * size  # 每个格子作为一个状态特征
        self.action_size = 4  # 上、下、左、右
        self.reset()
    
    def reset(self):
        """重置迷宫环境"""
        # 随机生成迷宫(0=空地, 1=墙壁)
        self.maze = np.random.choice([0, 1], size=(self.size, self.size), p=[0.8, 0.2])
        # 设置起点和终点
        self.start = (0, 0)
        self.goal = (self.size-1, self.size-1)
        self.position = self.start
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态表示"""
        state = np.zeros(self.state_size)
        idx = self.position[0] * self.size + self.position[1]
        state[idx] = 1  # 当前位置的one-hot编码
        return state
    
    def step(self, action):
        """
        执行动作
        :param action: 0=上, 1=下, 2=左, 3=右
        :return: (新状态, 奖励, 是否终止, 额外信息)
        """
        # 计算新位置
        new_pos = list(self.position)
        if action == 0: new_pos[0] = max(0, new_pos[0]-1)  # 上
        elif action == 1: new_pos[0] = min(self.size-1, new_pos[0]+1)  # 下
        elif action == 2: new_pos[1] = max(0, new_pos[1]-1)  # 左
        elif action == 3: new_pos[1] = min(self.size-1, new_pos[1]+1)  # 右
        
        # 检查是否撞墙
        if self.maze[new_pos[0], new_pos[1]] == 0:  # 空地
            self.position = tuple(new_pos)
        
        # 计算奖励
        reward = self._calculate_reward()
        done = self.position == self.goal
        return self._get_state(), reward, done, {}
    
    def _calculate_reward(self):
        """计算迷宫导航的奖励"""
        # 基础奖励
        base_reward = -0.1  # 每步小惩罚鼓励快速到达
        
        # 距离终点越近奖励越高
        distance = np.sqrt((self.goal[0]-self.position[0])**2 + 
                          (self.goal[1]-self.position[1])**2)
        distance_reward = 1.0 / (distance + 1)
        
        # 到达终点的大奖励
        goal_reward = 10.0 if self.position == self.goal else 0.0
        
        return base_reward + distance_reward + goal_reward