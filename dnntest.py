import numpy as np
import torch
from model import DNNModel
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Dataset


def test(input_signal,label_signal,model_path):


    dataset_tensor = torch.from_numpy(input_signal).float()  # 转换为float32张量
    label_tensor = torch.from_numpy(label_signal).float()  # 转换为float32张量
    tensor_dataset = TensorDataset(dataset_tensor, label_tensor)  # 确保标签是长整型
    test_loader = DataLoader(tensor_dataset, batch_size=64, shuffle=False)

    # 1. 加载模型
    checkpoint = torch.load(model_path)
    
    # 2. 重新创建模型结构（必须与训练时相同）
    input_size = 4
    hidden_sizes = [128, 64]
    output_size = 15
    model = DNNModel(input_size, hidden_sizes, output_size, dropout_rate=0.01)
    
    # 3. 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式（关闭dropout等）
    
    
    # 5. 进行预测
    predictions = []
    true_labels = []
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for batch_data, batch_labels in test_loader:
            # print(batch_data.shape)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            predictions.extend(predicted.numpy())
            true_labels.extend(batch_labels.numpy())
    
    return np.array(predictions), np.array(true_labels)


