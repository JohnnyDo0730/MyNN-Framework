"""
交叉熵 (Cross Entropy) 損失函數
"""

import torch

def softmax(logits):
    """
    計算 softmax 函數
    
    參數:
        logits: 輸入張量，形狀為 (C, N)，其中 C 是類別數，N 是樣本數
    
    返回:
        softmax 結果，與輸入形狀相同
    """
    # 防止溢出，減去最大值
    exp_logits = torch.exp(logits - torch.max(logits, dim=1, keepdim=True)[0])
    return exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)

def cross_entropy_loss(outputs, targets):
    """
    計算交叉熵損失及其梯度
    
    參數:
        outputs: 模型輸出，形狀為 (C, N)，其中 C 是類別數，N 是樣本數
        targets: 目標類別，形狀為 (N,)，其中 N 是樣本數
    
    返回:
        loss: 損失值
        grad: 損失對輸出的梯度
    """
    # 計算 softmax
    probs = softmax(outputs.T)  # 轉置輸出以適應計算

    # 創建 one-hot 編碼
    num_samples = targets.shape[0]
    num_classes = probs.shape[1]
    targets_one_hot = torch.zeros(num_samples, num_classes, device=outputs.device, dtype=outputs.dtype)
    targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)  # 使用 scatter_ 創建 one-hot 編碼

    # 計算交叉熵損失
    loss = -torch.sum(targets_one_hot * torch.log(probs + 1e-15)) / num_samples
    grad = probs - targets_one_hot
    return loss, grad 