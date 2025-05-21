"""
基础AI代理类
包含基本决策和行为能力
"""
from core.neural import EvolvableNetwork
import numpy as np

class BaseAgent:
    def __init__(self, state_size, action_size):
        """
        初始化代理
        :param state_size: 状态空间大小
        :param action_size: 动作空间大小
        """
        self.brain = EvolvableNetwork(
            input_size=state_size,
            hidden_sizes=[32, 32],  # 默认2层隐藏层
            output_size=action_size
        )
        self.fitness = 0.0  # 适应度分数
    
    def act(self, state):
        """
        根据当前状态选择动作
        :param state: 环境状态
        :return: 选择的动作
        """
        return self.brain.forward(state)
    
    def evaluate(self, reward):
        """
        评估并更新适应度
        :param reward: 获得的奖励
        """
        self.fitness += reward
    
    def clone(self):
        """
        创建代理的克隆体
        :return: 新代理实例
        """
        new_agent = BaseAgent(
            self.brain.layers[0]['weights'].shape[0],
            self.brain.layers[-1]['weights'].shape[1]
        )
        new_agent.brain.set_genome(self.brain.get_genome())
        return new_agent