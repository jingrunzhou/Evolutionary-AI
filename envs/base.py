"""
基础环境模拟器
提供进化所需的测试环境
"""
import numpy as np

class BaseEnvironment:
    def __init__(self, difficulty=1.0):
        """
        初始化环境
        :param difficulty: 环境难度系数
        """
        self.difficulty = difficulty
        self.state_size = 10  # 默认状态空间大小
        self.action_size = 3  # 默认动作空间大小
    
    def reset(self):
        """
        重置环境状态
        :return: 初始观察状态
        """
        return np.random.randn(self.state_size)
    
    def step(self, action):
        """
        执行一个时间步
        :param action: 代理执行的动作
        :return: (新状态, 奖励, 是否终止, 额外信息)
        """
        new_state = np.random.randn(self.state_size)
        reward = self._calculate_reward(action)
        done = random.random() < 0.01  # 1%概率终止
        info = {}
        return new_state, reward, done, info
    
    def _calculate_reward(self, action):
        """
        计算奖励函数
        :param action: 执行的动作
        :return: 奖励值
        """
        # 基础奖励 + 动作质量奖励
        base_reward = 0.1 * self.difficulty
        action_quality = 1.0 - np.abs(action).mean()
        return base_reward * action_quality