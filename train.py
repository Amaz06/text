
import time
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Dataset
import torch
import matplotlib.pyplot as plt
from model import DNNModel
import torch.nn as nn
from datetime import datetime

def train(dataset,label,lr = 0.0001,num_epochs = 10,dropout_rate=0.01):
    dataset_tensor = torch.from_numpy(dataset).float()  # 转换为float32张量
    label_tensor = torch.from_numpy(label).float()  # 转换为float32张量
    tensor_dataset = TensorDataset(dataset_tensor, label_tensor)  # 确保标签是长整型
    train_loader = DataLoader(tensor_dataset, batch_size=64, shuffle=True)
    
    loss_history = []

    input_size = 4
    hidden_sizes = [128, 64]
    output_size = 15


    model = DNNModel(input_size, hidden_sizes, output_size,dropout_rate=dropout_rate)
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()  # 切换到训练模式
        running_loss = 0.0
        for batch_dataset, batch_labels in train_loader:
            # print(batch_dataset)
            # print(batch_labels)
            optimizer.zero_grad()  # 清空梯度
            outputs = model(batch_dataset)  # 计算模型输出
            loss = criterion(outputs, batch_labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            
            running_loss += loss.item()  # 累加损失
        epoch_loss = running_loss / len(train_loader)  # 计算平均损失
        loss_history.append(epoch_loss)  # 将平均损失添加到历史记录中    
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")



    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"models/model_{timestamp}_lr{lr}.pth"
    end_time = time.time()
    train_time = end_time - start_time
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'training_time': train_time,
        'lr': lr,
        'num_epochs': num_epochs
    }, model_filename)
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
    plt.title(f'Training Loss Over Epochs in lr = {lr} ')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f"loss_gragh_{lr}.png")
    print(f"Training time: {train_time:.2f} seconds")