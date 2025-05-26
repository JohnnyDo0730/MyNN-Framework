"""
MNIST 訓練示例 - 使用 torchvision
"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

from src.layers import myModule, myLinear, myLeakyReLU, myDropout
from src.optimizers import Adam
from src.losses import cross_entropy_loss

# 定義神經網路模型
class myNet(myModule):
    """
    神經網路模型，實現 MNIST 分類
    """
    def __init__(self):
        """
        初始化神經網路模型
        """
        super().__init__()
        # 定義網路結構
        self.fc1 = myLinear(28 * 28, 512)
        self.relu1 = myLeakyReLU()
        self.dropout1 = myDropout(0.2)
        self.fc2 = myLinear(512, 128)
        self.relu2 = myLeakyReLU()
        self.fc3 = myLinear(128, 10)

    def forward(self, x):
        """
        前向傳播
        
        參數:
            x: 輸入張量，形狀為 (N, 1, 28, 28)，其中 N 是批量大小
        
        返回:
            輸出張量，形狀為 (10, N)
        """
        # 輸入維度從 PyTorch batch 計算轉換成 chain rule 公式形式
        x = torch.flatten(x, 1).T  # [batch, gray, x, y] => [x*y, batch]
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        x = self.dropout1.forward(x)
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        x = self.fc3.forward(x)
        return x

    def backward(self, error):
        """
        反向傳播
        
        參數:
            error: 誤差張量
        
        返回:
            梯度張量
        """
        # 輸入維度從 PyTorch batch 計算轉換成 chain rule 公式形式
        error = error.T
        error = self.fc3.backward(error)
        error = self.relu2.backward(error)
        error = self.fc2.backward(error)
        error = self.relu1.backward(error)
        error = self.dropout1.backward(error)
        error = self.fc1.backward(error)
        return error

def main():
    # 設定參數
    batch_size = 128
    epochs = 10
    learning_rate = 0.002
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 定義轉換
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 載入 MNIST 資料集
    train_ds = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # 創建資料載入器
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # 創建模型
    model = myNet().to(device)
    
    # 創建優化器
    optimizer = Adam(learning_rate)

    # 訓練模型
    loss_history = []
    for epoch in range(1, epochs + 1):
        model.set_train_state(True)
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 將資料移到指定設備
            data, target = data.to(device), target.to(device)
            
            # 前向傳播
            outputs = model.forward(data)
            
            # 計算損失
            loss, grad = cross_entropy_loss(outputs, target)
            epoch_loss += loss.item()
            batch_count += 1
            
            # 反向傳播
            model.backward(grad)
            
            # 更新參數
            optimizer.step(model.parameters())
            
            # 每 10 個批次顯示一次
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}: [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                      f"({100. * batch_idx / len(train_loader):.0f}%) Loss: {loss.item():.6f}")
                loss_history.append(loss.item())
        
        # 計算平均損失
        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch} 平均損失: {avg_loss:.6f}")
        
        # 測試模型
        model.set_train_state(False)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model.forward(data)
                _, predicted = torch.max(outputs.T, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch} 測試準確率: {accuracy:.2f}%")
    
    # 繪製損失曲線
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Iterations (x10)")
    plt.ylabel("Loss")
    plt.savefig("loss_curve.png")
    plt.show()

if __name__ == "__main__":
    main() 