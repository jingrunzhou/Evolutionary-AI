"""
可进化神经网络基础实现
支持动态结构调整
"""
import numpy as np

class EvolvableNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, use_lstm=False):
        """
        初始化神经网络
        :param input_size: 输入层大小
        :param hidden_sizes: 隐藏层大小列表
        :param output_size: 输出层大小
        :param use_lstm: 是否使用LSTM记忆层
        """
        self.layers = []
        self.use_lstm = use_lstm
        sizes = [input_size] + hidden_sizes + [output_size]
        
        # 添加LSTM层
        if use_lstm:
            self.lstm_size = 32  # LSTM隐藏层大小
            self.lstm_state = None
            self.lstm_cell = {
                'Wf': np.random.randn(input_size + self.lstm_size, self.lstm_size) * 0.1,
                'Wi': np.random.randn(input_size + self.lstm_size, self.lstm_size) * 0.1,
                'Wo': np.random.randn(input_size + self.lstm_size, self.lstm_size) * 0.1,
                'Wc': np.random.randn(input_size + self.lstm_size, self.lstm_size) * 0.1
            }
            sizes[0] = self.lstm_size  # LSTM输出作为第一层输入
        
        # 初始化权重和偏置
        for i in range(len(sizes)-1):
            # Xavier初始化
            limit = np.sqrt(6.0 / (sizes[i] + sizes[i+1]))
            weights = np.random.uniform(-limit, limit, (sizes[i], sizes[i+1]))
            biases = np.zeros(sizes[i+1])
            self.layers.append({'weights': weights, 'biases': biases})
    
    def forward(self, x, prev_state=None):
        """
        带记忆的前向传播
        :param x: 输入数据
        :param prev_state: 前一个LSTM状态(h, c)
        :return: (网络输出, 新状态)
        """
        if self.use_lstm:
            # LSTM处理
            h_prev, c_prev = prev_state if prev_state else (
                np.zeros(self.lstm_size), 
                np.zeros(self.lstm_size)
            )
            
            # 拼接输入和前一隐藏状态
            combined = np.concatenate([x, h_prev])
            
            # LSTM门计算
            f = sigmoid(np.dot(combined, self.lstm_cell['Wf']))
            i = sigmoid(np.dot(combined, self.lstm_cell['Wi']))
            o = sigmoid(np.dot(combined, self.lstm_cell['Wo']))
            c_tilde = np.tanh(np.dot(combined, self.lstm_cell['Wc']))
            
            # 更新细胞状态和隐藏状态
            c = f * c_prev + i * c_tilde
            h = o * np.tanh(c)
            
            x = h  # 使用LSTM输出作为下一层输入
            new_state = (h, c)
        else:
            new_state = None
        
        for layer in self.layers:
            x = np.dot(x, layer['weights']) + layer['biases']
            x = np.tanh(x)  # 使用tanh激活函数
        
        return x, new_state

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

    def get_genome(self):
        """获取网络基因组(用于遗传算法)"""
        genome = []
        for layer in self.layers:
            genome.extend(layer['weights'].flatten())
            genome.extend(layer['biases'])
        return np.array(genome)
    
    def set_genome(self, genome):
        """从基因组设置网络参数"""
        ptr = 0
        for layer in self.layers:
            w_shape = layer['weights'].shape
            w_size = w_shape[0] * w_shape[1]
            layer['weights'] = genome[ptr:ptr+w_size].reshape(w_shape)
            ptr += w_size
            
            b_size = layer['biases'].size
            layer['biases'] = genome[ptr:ptr+b_size]
            ptr += b_size
    
    def mutate_structure(self, mutation_rate=0.1):
        """
        动态调整网络结构
        :param mutation_rate: 结构变异概率
        """
        if random.random() < mutation_rate:
            # 随机选择变异类型
            mutation_type = random.choice(['add', 'remove', 'modify'])
            
            if mutation_type == 'add' and len(self.layers) < 5:  # 限制最大层数
                # 在随机位置添加新层
                insert_pos = random.randint(1, len(self.layers)-1)
                new_size = random.randint(16, 64)
                
                # 初始化新层权重
                prev_size = self.layers[insert_pos-1]['weights'].shape[1]
                next_size = self.layers[insert_pos]['weights'].shape[0]
                weights = np.random.randn(prev_size, new_size) * 0.1
                biases = np.zeros(new_size)
                
                # 插入新层并调整相邻层
                self.layers.insert(insert_pos, {'weights': weights, 'biases': biases})
                self.layers[insert_pos+1]['weights'] = np.random.randn(new_size, next_size) * 0.1
            
            elif mutation_type == 'remove' and len(self.layers) > 1:
                # 随机移除一层(保留输入输出层)
                remove_pos = random.randint(1, len(self.layers)-2)
                
                # 调整相邻层连接
                prev_size = self.layers[remove_pos-1]['weights'].shape[1]
                next_size = self.layers[remove_pos+1]['weights'].shape[0]
                self.layers[remove_pos-1]['weights'] = np.random.randn(prev_size, next_size) * 0.1
                
                del self.layers[remove_pos]
            
            elif mutation_type == 'modify':
                # 随机修改一层的大小
                modify_pos = random.randint(1, len(self.layers)-1)
                new_size = random.randint(16, 64)
                
                # 调整权重矩阵
                if modify_pos > 0:  # 不是第一层
                    prev_size = self.layers[modify_pos-1]['weights'].shape[1]
                    self.layers[modify_pos-1]['weights'] = np.random.randn(prev_size, new_size) * 0.1
                
                if modify_pos < len(self.layers)-1:  # 不是最后一层
                    next_size = self.layers[modify_pos+1]['weights'].shape[0]
                    self.layers[modify_pos]['weights'] = np.random.randn(new_size, next_size) * 0.1
                
                self.layers[modify_pos]['biases'] = np.zeros(new_size)