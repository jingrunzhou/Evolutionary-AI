"""
进化主循环实现
控制AI种群的进化过程
"""
import numpy as np
from typing import List
from agents.base_agent import BaseAgent

class EvolutionEngine:
    def __init__(self, population_size=50, elitism=0.2):
        """
        初始化进化引擎
        :param population_size: 种群大小
        :param elitism: 精英保留比例
        """
        self.population_size = population_size
        self.elitism = elitism
        self.population: List[BaseAgent] = []
    
    def initialize_population(self, state_size, action_size):
        """
        初始化AI种群
        :param state_size: 状态空间大小
        :param action_size: 动作空间大小
        """
        self.population = [
            BaseAgent(state_size, action_size) 
            for _ in range(self.population_size)
        ]
    
    def evolve(self):
        """
        执行一代进化
        1. 评估适应度
        2. 选择优秀个体
        3. 交叉和变异
        :return: 新一代种群
        """
        # 按适应度排序
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        
        # 精英保留
        elite_size = int(self.elitism * self.population_size)
        elites = sorted_pop[:elite_size]
        
        # 选择父代
        parents = self._select_parents(sorted_pop)
        
        # 生成后代
        offspring = []
        for i in range(0, len(parents), 2):
            if i+1 >= len(parents):
                break
            child1, child2 = self._crossover(parents[i], parents[i+1])
            offspring.extend([child1, child2])
        
        # 确保种群大小不变
        next_gen = elites + offspring[:self.population_size - elite_size]
        self.population = next_gen
        return next_gen
    
    def _select_parents(self, population):
        """轮盘赌选择父代"""
        fitnesses = np.array([agent.fitness for agent in population])
        probs = fitnesses / fitnesses.sum()
        return np.random.choice(population, size=len(population), p=probs)
    
    def _crossover(self, parent1, parent2):
        """单点交叉产生后代"""
        child1 = parent1.clone()
        child2 = parent2.clone()
        # 这里可以添加更复杂的交叉逻辑
        return child1, child2