import torch
import torch.nn as nn
import torch.nn.functional as F





class RNN8Class(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=8,train_model = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.train_model = train_model
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # 8类输出

    def forward(self, x):
        print("x.shape:", x.shape)
        # x 形状: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()

        # 初始化隐藏状态（h0）和 RNN 输出（初始为全零）
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        prev_mid_bit = torch.zeros(batch_size, 1).to(x.device)


        outputs = []

        if self.train_model :
            out, _ = self.rnn(x, h0)
            logits = self.fc(out)  # shape: (batch, seq_len, num_classes)
            return logits  # 可以直接用于 loss 计算
        
        else :
            for t in range(seq_len):
                signal_input = x[:, t, :]  # (batch, 6)
                print(signal_input)
                rnn_input = torch.cat([prev_mid_bit,signal_input], dim=-1).unsqueeze(1) # (batch, 1, 7)
                print(rnn_input)
                
                out, h0 = self.rnn(rnn_input, h0)  # (batch, 1, hidden_size)
                logits = self.fc(out.squeeze(1))  # (batch, num_classes)
                outputs.append(logits)

                # 分类预测 -> 3位二进制 -> 取中间bit作为下一个输入
                pred_class = torch.argmax(logits, dim=-1)  # (batch,)
                pred_bits = F.one_hot(pred_class, num_classes=8)  # (batch, 8)
                pred_3bit = ((pred_class[:, None] >> torch.tensor([2, 1, 0]).to(x.device)) & 1).float()  # (batch, 3)
                prev_mid_bit = pred_3bit[:, 1:2]  # 取中间位 (bit[1]) -> shape: (batch, 1)

            
            return torch.stack(outputs, dim=1)
        



class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5):
        super(DNNModel, self).__init__()
        
        # 确保hidden_sizes列表长度至少为2，以支持至少两个隐藏层
        if len(hidden_sizes) < 2:
            raise ValueError("hidden_sizes list must contain at least two elements.")
        
        # 定义神经网络的各层
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])  # 第一层
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])  # 第二层
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)  # 输出层
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 激活函数
        self.relu = nn.ReLU()
 
    def forward(self, x):
        # 前向传播
        x = self.relu(self.fc1(x))  # 第一层 -> ReLU 激活
        x = self.relu(self.fc2(x))  # 第二层 -> ReLU 激活
        x = self.fc3(x)  # 输出层 (没有激活函数，因为输出是一个类别概率)
        return x