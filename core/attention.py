"""
注意力机制模块
为神经网络添加注意力能力
"""
import numpy as np

class AttentionMechanism:
    def __init__(self, input_size, attention_size):
        """
        初始化注意力机制
        :param input_size: 输入维度
        :param attention_size: 注意力空间大小
        """
        # 注意力权重参数
        self.W = np.random.randn(input_size, attention_size) * 0.1
        self.U = np.random.randn(attention_size, 1) * 0.1
        
    def forward(self, inputs):
        """
        计算注意力权重
        :param inputs: 输入序列 [seq_len, input_size]
        :return: 加权后的特征向量
        """
        # 计算注意力分数
        scores = np.dot(np.tanh(np.dot(inputs, self.W)), self.U)
        # Softmax归一化
        alpha = np.exp(scores) / np.sum(np.exp(scores), axis=0)
        # 加权求和
        context = np.sum(inputs * alpha, axis=0)
        return context