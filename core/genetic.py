"""
遗传算法核心实现
包含选择、交叉、变异等基本操作
"""
import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size, gene_length):
        """
        初始化遗传算法
        :param population_size: 种群大小
        :param gene_length: 基因长度
        """
        self.population_size = population_size
        self.gene_length = gene_length
        self.population = self._init_population()
    
    def _init_population(self):
        """初始化种群"""
        return np.random.rand(self.population_size, self.gene_length)
    
    def selection(self, fitness_scores):
        """
        选择操作(轮盘赌选择)
        :param fitness_scores: 适应度分数数组
        :return: 选择后的新种群
        """
        probabilities = fitness_scores / fitness_scores.sum()
        selected_indices = np.random.choice(
            len(self.population), 
            size=self.population_size,
            p=probabilities
        )
        return self.population[selected_indices]
    
    def crossover(self, parent1, parent2, crossover_rate=0.8):
        """
        单点交叉操作
        :param parent1: 父代1
        :param parent2: 父代2 
        :param crossover_rate: 交叉概率
        :return: 两个子代
        """
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, self.gene_length-1)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    
    def mutation(self, individual, mutation_rate=0.01):
        """
        基因突变
        :param individual: 个体基因
        :param mutation_rate: 突变概率
        :return: 突变后的个体
        """
        mask = np.random.random(self.gene_length) < mutation_rate
        individual[mask] = np.random.rand(mask.sum())
        return individual