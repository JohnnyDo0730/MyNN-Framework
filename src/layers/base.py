"""
神經網路模組基礎類
"""

import torch

class myModule:
    """
    神經網路模組基礎類，所有層都繼承自此類
    """
    def __init__(self):
        self.train_state = True
        self.sub_layers = []

    def set_train_state(self, state):
        """設置訓練狀態"""
        self.train_state = state
        for layer in self.sub_layers:
            layer.set_train_state(state)

    def forward(self, x):
        """前向傳播"""
        raise NotImplementedError("子類必須實現此方法！")

    def backward(self, error):
        """反向傳播"""
        raise NotImplementedError("子類必須實現此方法！")

    def get_layers(self):
        """獲取所有子層"""
        return self.sub_layers

    def to(self, device):
        """將模組移動到指定設備"""
        for layer in self.sub_layers:
            layer.to(device)
        return self

    def parameters(self):
        """獲取所有可訓練參數"""
        params = []
        for layer in self.sub_layers:
            params.extend(layer.parameters())
        return params

    def __setattr__(self, name, value):
        """設置屬性時自動註冊子層"""
        if isinstance(value, myModule):  # 如果是 myModule 類的實例
            if not hasattr(self, 'sub_layers'): # Cursor added this line
                super().__setattr__('sub_layers', [])
            if value not in self.sub_layers:  # 避免重複加入
                self.sub_layers.append(value)
        super().__setattr__(name, value)  # 預設方法、建立任何子物件都要執行 