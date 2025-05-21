"""
进化过程可视化工具
"""
import matplotlib.pyplot as plt
import numpy as np

class EvolutionVisualizer:
    @staticmethod
    def plot_fitness_history(fitness_history):
        """
        绘制适应度变化曲线
        :param fitness_history: 各代的平均适应度历史
        """
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history, label='平均适应度')
        plt.xlabel('代数')
        plt.ylabel('适应度')
        plt.title('进化过程适应度变化')
        plt.legend()
        plt.grid()
        plt.show()
    
    @staticmethod
    def plot_population_diversity(population):
        """
        绘制种群基因多样性热力图
        :param population: 当前种群
        """
        genomes = [agent.brain.get_genome() for agent in population]
        corr_matrix = np.corrcoef(genomes)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title('种群基因相似度矩阵')
        plt.xlabel('个体索引')
        plt.ylabel('个体索引')
        plt.show()