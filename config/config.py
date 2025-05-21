"""
参数配置系统
使用YAML管理所有实验参数
"""
import yaml
from pathlib import Path

class ConfigManager:
    def __init__(self, config_path="config/experiment.yml"):
        """
        初始化配置管理器
        :param config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get(self, key, default=None):
        """
        获取配置参数
        :param key: 参数键名(支持点分隔)
        :param default: 默认值
        :return: 配置值
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
            if value == {}:
                return default
        return value