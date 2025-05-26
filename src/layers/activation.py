"""
激活函數層模組
"""

import torch
from .base import myModule

class myReLU(myModule):
    """
    ReLU 激活函數：f(x) = max(0, x)
    """
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self, x):
        """前向傳播"""
        self.x = x
        return torch.maximum(torch.tensor(0.0, dtype=x.dtype).to(x.device), x)

    def backward(self, error):
        """反向傳播"""
        # ReLU(x)' = x>0 ? 1:0
        return error * (self.x > 0).float()

    def parameters(self):
        """返回可訓練參數（無參數）"""
        return []


class myLeakyReLU(myModule):
    """
    Leaky ReLU 激活函數：f(x) = x if x > 0 else alpha * x
    """
    def __init__(self, alpha=0.01):
        super().__init__()
        self.x = None
        self.alpha = alpha  # 控制負區域的斜率

    def forward(self, x):
        """前向傳播"""
        self.x = x
        return torch.where(x > 0, x, self.alpha * x)

    def backward(self, error):
        """反向傳播"""
        # Leaky ReLU(x)' = x > 0 ? 1 : alpha
        return error * torch.where(self.x > 0, torch.ones_like(self.x), self.alpha * torch.ones_like(self.x))

    def parameters(self):
        """返回可訓練參數（無參數）"""
        return []


class mySigmoid(myModule):
    """
    Sigmoid 激活函數：f(x) = 1 / (1 + exp(-x))
    使用絕對數值、歸一化到 [0,1] 時使用
    """
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        """前向傳播"""
        self.output = torch.sigmoid(x)
        return self.output

    def backward(self, error):
        """反向傳播"""
        # sigmoid(x)' = sigmoid(x) * (1 - sigmoid(x))
        return error * self.output * (1 - self.output)

    def parameters(self):
        """返回可訓練參數（無參數）"""
        return []


class myTanh(myModule):
    """
    Tanh 激活函數：f(x) = tanh(x)
    使用差分、標準化到 [-1,1] 時使用
    """
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        """前向傳播"""
        self.output = torch.tanh(x)
        return self.output

    def backward(self, error):
        """反向傳播"""
        # tanh(x)' = 1 - tanh^2(x)
        return error * (1 - self.output ** 2)

    def parameters(self):
        """返回可訓練參數（無參數）"""
        return [] 