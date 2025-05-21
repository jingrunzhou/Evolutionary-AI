"""
进化过程记录器
记录并保存进化过程数据
"""
import json
import time
from pathlib import Path

class EvolutionLogger:
    def __init__(self, log_dir="logs"):
        """
        初始化记录器
        :param log_dir: 日志目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"evolution_{int(time.time())}.json"
        self.data = []
    
    def log_generation(self, gen, population, metrics):
        """
        记录一代进化数据
        :param gen: 代数
        :param population: 种群
        :param metrics: 评估指标
        """
        record = {
            "generation": gen,
            "timestamp": time.time(),
            "best_fitness": max([ind.fitness for ind in population]),
            "avg_fitness": sum([ind.fitness for ind in population])/len(population),
            "metrics": metrics
        }
        self.data.append(record)
    
    def save(self):
        """保存日志到文件"""
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=2)