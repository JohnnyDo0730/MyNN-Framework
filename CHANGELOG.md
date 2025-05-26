# 版本歷史

## v0.1.0 (當前版本)
- 模組化架構完成
- 使用 pipenv 進行依賴管理
- 使用 torchvision 載入 MNIST 數據集

## v0.0.6 (先前版本)
- 所有層都繼承自 myModule 類
- 將註冊機制改為遞迴，移除 isLayer、isWeightLayer 布林值
- 將 .to 與 .set_train_state 方法改為遞迴
- 更改 dropout 邏輯(判斷自己的參數)
- 移除 AF 抽象類
- 新增 myModule 更新參數的回傳邏輯，更改 Optimizer 遍歷邏輯

## v0.0.5
- Net 改成繼承 Module 抽象類
- 覆寫 __setattr__ 實現自動註冊 layers，添加 is_Layer、is_WeightLayer 布林值
- 抽象化 Activation Function 類
- 根據 is_WeightLayer 更改 Optimizer 類邏輯
- 將 batch 計算邏輯移到 myLinear 內
- 新增 sigmoid、tanh、MSELoss 等時序相關方法

## v0.0.4
- 維度反轉操作移到模型內，使操作與 PyTorch 對齊
- 將 tensor 精度從 float32 改成 float64(與 np 相同)改善精度問題
- 新增 leaky ReLU、初始化使用 uniform 改善梯度消失問題
- 新增 Adam optimizer

## v0.0.3
- numpy 運算改成 tensor 運算，適配 GPU 訓練
- 所有層結構、神經網路架構和優化相關結構重新實現為 PyTorch 版本

## v0.0.2
- 將梯度更新抽離 backward、整合到 optimizer 抽象類
- loss 計算適配 batch 更新
- 實現 Cross Entropy Loss 計算
- 新增 Momentum optimizer

## v0.0.1
- 基本神經網路架構實現
- 實現全連接層 (myLinear)
- 實現 dropout 層
- 實現 ReLU 激活函數
- backward 中直接更新權重 