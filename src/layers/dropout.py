"""
Dropout 層模組
"""

import torch
from .base import myModule

class myDropout(myModule):
    """
    Dropout 層，用於防止過擬合
    在訓練時隨機將一部分神經元輸出置為0
    """
    def __init__(self, p=0.5, dtype=torch.float64):
        super().__init__()
        self.p = p  # 丟棄概率
        self.mask = None
        self.dtype = dtype

    def forward(self, x):
        """前向傳播"""
        if not self.train_state:
            return x  # 測試時直接返回輸入

        # 隨機生成遮罩
        self.mask = torch.rand_like(x, dtype=self.dtype) > self.p
        # 遮罩應用並進行縮放
        return x * self.mask / (1 - self.p)

    def backward(self, error):
        """反向傳播"""
        # 只保留激活的部分
        return error * self.mask

    def parameters(self):
        """返回可訓練參數（無參數）"""
        return [] 