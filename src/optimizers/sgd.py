"""
隨機梯度下降 (SGD) 優化器
"""

from .base import Optimizer

class SGD(Optimizer):
    """
    隨機梯度下降優化器
    
    更新規則：
        param = param - learning_rate * grad
    """
    def __init__(self, learning_rate):
        """
        初始化 SGD 優化器
        
        參數:
            learning_rate: 學習率
        """
        super().__init__(learning_rate)

    def update(self, param, grad):
        """
        更新參數
        
        參數:
            param: 參數
            grad: 梯度
        """
        param -= self.learning_rate * grad 