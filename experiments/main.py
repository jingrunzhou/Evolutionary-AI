"""
主实验脚本
控制整个进化实验流程
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))



from core.evolution import EvolutionEngine
from envs.challenges.maze import MazeEnvironment
from utils.metrics import FitnessEvaluator
from utils.visualization import EvolutionVisualizer
import numpy as np

def run_experiment():
    """运行完整进化实验"""
    # 初始化环境
    env = MazeEnvironment(size=10)
    
    # 初始化进化引擎
    evolution = EvolutionEngine(population_size=50)
    evolution.initialize_population(
        state_size=env.state_size,
        action_size=env.action_size
    )
    
    # 进化参数
    generations = 100
    fitness_history = []
    
    # 进化循环
    for gen in range(generations):
        # 评估种群
        for agent in evolution.population:
            agent.fitness = FitnessEvaluator.evaluate_agent(agent, env)
        
        # 记录统计信息
        avg_fitness = np.mean([agent.fitness for agent in evolution.population])
        fitness_history.append(avg_fitness)
        print(f"Generation {gen}: Avg Fitness = {avg_fitness:.2f}")
        
        # 执行进化
        evolution.evolve()
    
    # 可视化结果
    EvolutionVisualizer.plot_fitness_history(fitness_history)
    EvolutionVisualizer.plot_population_diversity(evolution.population)

if __name__ == "__main__":
    run_experiment()