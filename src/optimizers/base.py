"""
優化器基礎類
"""

class Optimizer:
    """
    優化器基礎類，所有優化器都繼承自此類
    """
    def __init__(self, learning_rate):
        """
        初始化優化器
        
        參數:
            learning_rate: 學習率
        """
        self.learning_rate = learning_rate
        self.t = 0  # 時間步長，用於某些優化器的計算

    def step(self, parameters_and_gradients):
        """
        執行一步優化
        
        參數:
            parameters_and_gradients: 參數和梯度的列表，格式為 [[param1, grad1], [param2, grad2], ...]
        """
        self.t += 1
        for param, grad in parameters_and_gradients:
            self.update(param, grad)

    def update(self, param, grad):
        """
        更新參數
        
        參數:
            param: 參數
            grad: 梯度
        """
        raise NotImplementedError("子類必須實現此方法！") 