"""
動量 (Momentum) 優化器
"""

import torch
from .base import Optimizer

class Momentum(Optimizer):
    """
    動量優化器
    
    更新規則：
        v = momentum * v + learning_rate * grad
        param = param - v
    """
    def __init__(self, learning_rate, momentum=0.9):
        """
        初始化 Momentum 優化器
        
        參數:
            learning_rate: 學習率
            momentum: 動量係數，默認為 0.9
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}  # 儲存各參數的速度

    def update(self, param, grad):
        """
        更新參數
        
        參數:
            param: 參數
            grad: 梯度
        """
        # 初始化速度
        if param not in self.velocities:
            self.velocities[param] = torch.zeros_like(param, dtype=grad.dtype)

        # 更新速度
        v = self.velocities[param]
        v = self.momentum * v + self.learning_rate * grad
        self.velocities[param] = v

        # 更新參數
        param -= v 