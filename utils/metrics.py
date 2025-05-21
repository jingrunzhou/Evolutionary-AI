"""
高级适应度评估系统
包含多种评估指标
"""
import numpy as np

class FitnessEvaluator:
    @staticmethod
    def evaluate_agent(agent, env, episodes=10):
        """
        全面评估代理在环境中的表现
        :param agent: 要评估的AI代理
        :param env: 测试环境
        :param episodes: 评估回合数
        :return: 综合适应度分数
        """
        total_reward = 0.0
        steps_to_goal = []
        success_rate = 0.0
        
        for _ in range(episodes):
            state = env.reset()
            done = False
            steps = 0
            episode_reward = 0.0
            
            while not done and steps < 1000:  # 防止无限循环
                action = agent.act(state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                steps += 1
            
            total_reward += episode_reward
            if done:  # 成功到达终点
                steps_to_goal.append(steps)
                success_rate += 1.0
        
        # 计算综合适应度
        avg_reward = total_reward / episodes
        avg_steps = np.mean(steps_to_goal) if steps_to_goal else 1000
        success_rate /= episodes
        
        # 加权综合适应度
        fitness = 0.6 * avg_reward + 0.3 * (1 - avg_steps/1000) + 0.1 * success_rate
        return fitness


class MultiObjectiveEvaluator:
    """
    多目标优化评估系统
    评估AI代理在多个目标上的表现
    """
    @staticmethod
    def evaluate(agent, env, objectives):
        """
        多目标评估
        :param agent: 要评估的AI代理
        :param env: 测试环境
        :param objectives: 目标配置字典
        :return: 各目标得分字典
        """
        results = {}
        
        # 评估每个目标
        for obj_name, obj_config in objectives.items():
            if obj_name == "navigation":
                results[obj_name] = FitnessEvaluator.evaluate_agent(agent, env)
            elif obj_name == "energy_efficiency":
                results[obj_name] = agent.energy_usage / env.steps if env.steps > 0 else 0
            elif obj_name == "exploration":
                results[obj_name] = len(env.visited_states) / env.total_states
        
        return results

    @staticmethod
    def aggregate_scores(scores, weights):
        """
        聚合多个目标得分
        :param scores: 各目标得分字典
        :param weights: 各目标权重字典
        :return: 综合适应度分数
        """
        weighted_sum = 0.0
        for obj_name, score in scores.items():
            weighted_sum += score * weights.get(obj_name, 1.0)
        return weighted_sum