# MyNNFramework (v0.1.0)

自定義神經網路框架，用於學習和實驗目的。

## 環境設置

本專案使用 pipenv 進行依賴管理。請按照以下步驟設置環境：

### 安裝 pipenv

如果尚未安裝 pipenv，請先安裝：

```bash
pip install pipenv
```

### 設置專案環境

```bash
# 初始化環境
pipenv install

# 進入虛擬環境
pipenv shell
```

## 執行 MNIST 訓練示例

在虛擬環境中，執行以下命令來訓練 MNIST 模型：

```bash
python train_mnist.py
```

## 專案結構

```
MyNNFramework/
├── src/                      # 源代碼目錄
│   ├── layers/               # 神經網路層
│   │   ├── base.py           # 基礎模組類
│   │   ├── linear.py         # 線性層
│   │   ├── dropout.py        # Dropout 層
│   │   └── activation.py     # 激活函數層
│   ├── optimizers/           # 優化器
│   │   ├── base.py           # 基礎優化器類
│   │   ├── sgd.py            # SGD 優化器
│   │   ├── momentum.py       # Momentum 優化器
│   │   └── adam.py           # Adam 優化器
│   └── losses/               # 損失函數
│       ├── mse.py            # 均方誤差損失
│       └── cross_entropy.py  # 交叉熵損失
├── train_mnist.py            # MNIST 訓練示例
└── CHANGELOG.md              # 版本變更記錄
```

## 功能特點

- 基於 PyTorch 的張量操作
- 支援 GPU 訓練
- 模組化設計，易於擴展
- 實現多種優化器：SGD、Momentum、Adam
- 實現多種損失函數：MSE、Cross Entropy
- 實現多種激活函數：ReLU、Leaky ReLU、Sigmoid、Tanh

## 依賴套件

主要依賴：
- torch: PyTorch 深度學習框架
- torchvision: 用於數據集載入 (僅用於示例程式)
- matplotlib: 用於繪製訓練曲線

## 版本歷史

目前版本: v0.1.0 (模組化完成)

詳細版本歷史請參考 [CHANGELOG.md](CHANGELOG.md) 文件。 