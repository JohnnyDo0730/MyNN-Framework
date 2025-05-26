"""
均方誤差 (MSE) 損失函數
"""

import torch

def mse_loss(outputs, targets):
    """
    計算均方誤差損失及其梯度
    
    參數:
        outputs: 模型輸出
        targets: 目標值
    
    返回:
        loss: 損失值
        grad: 損失對輸出的梯度
    """
    error = outputs - targets
    loss = torch.mean(torch.square(error)) / 2
    grad = error
    return loss, grad 