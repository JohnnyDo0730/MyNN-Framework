"""
線性層模組
"""

import torch
from .base import myModule

class myLinear(myModule):
    """
    線性層，實現全連接操作：y = Wx + b
    """
    def __init__(self, input_size, output_size, device='cpu', dtype=torch.float64):
        super().__init__()
        # 初始化參數
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dtype = dtype

        # 用于儲存梯度
        self.x = None
        self.dW = None
        self.db = None

        # 初始化權重並移到設備
        # 使用均勻分布初始化權重，範圍為 [-1, 1]
        self.W = 2 * torch.rand(output_size, input_size, device=self.device, dtype=self.dtype) - 1
        self.b = torch.zeros(output_size, 1, device=self.device, dtype=self.dtype)

    def forward(self, x):
        """前向傳播"""
        self.x = x.to(self.device, dtype=self.dtype)  # 儲存本次計算結果
        return torch.matmul(self.W, self.x) + self.b

    def backward(self, error):
        """反向傳播"""
        # 計算batch_size
        batch_size = error.size(1)

        # 計算自己的更新梯度 (將batch計算邏輯移到此處)
        self.dW = torch.matmul(error, self.x.T) / batch_size
        self.db = torch.sum(error, dim=1, keepdim=True) / batch_size

        # 梯度傳遞
        gradient_transfer = torch.matmul(self.W.T, error)
        return gradient_transfer

    def to(self, device):
        """將參數轉移到設備"""
        self.device = device
        self.W = self.W.to(device)
        self.b = self.b.to(device)
        return self

    def parameters(self):
        """返回可訓練參數"""
        return [[self.W, self.dW], [self.b, self.db]] 