"""
Adam 優化器
"""

import torch
from .base import Optimizer

class Adam(Optimizer):
    """
    Adam 優化器
    
    更新規則：
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        param = param - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    """
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        初始化 Adam 優化器
        
        參數:
            learning_rate: 學習率
            beta1: 一階矩估計的指數衰減率，默認為 0.9
            beta2: 二階矩估計的指數衰減率，默認為 0.999
            epsilon: 防止除零的小常數，默認為 1e-8
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # 一階矩
        self.v = {}  # 二階矩

    def update(self, param, grad):
        """
        更新參數
        
        參數:
            param: 參數
            grad: 梯度
        """
        # 初始化一階矩和二階矩
        if param not in self.m:
            self.m[param] = torch.zeros_like(param)
            self.v[param] = torch.zeros_like(param)

        # 更新一階矩和二階矩
        self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
        self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * grad ** 2

        # 偏置修正
        m_hat = self.m[param] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param] / (1 - self.beta2 ** self.t)

        # 更新參數
        param -= self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon) 