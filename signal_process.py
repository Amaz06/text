import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# import torch.nn as nn
# import torch
# from torch.utils.data import TensorDataset
# from torch.utils.data import DataLoader, Dataset
# import torch.optim as optim



Nos = 10
def gen_ook(ndata):
    """
    ndata: number of data
    """
    ook_data = np.random.randint(2, size=ndata)#A binary sequence of ndata bits, scaled to [0, 1]
    return ook_data

def oversampling(sequence, Nos = 10, reverse = False,graph = False):
    '''
        reverse : If is true, when the signal length is even, the signal position will be in the middle and rear part
    '''
    if reverse:
        front = Nos//2;
        behind = (Nos- 1)//2
    else :
        behind = Nos//2;
        front = (Nos - 1)//2

    oversampled = []
    for num in sequence:
        # 前面插入4个0
        oversampled.extend([0] * front)
        # 添加原始码元
        oversampled.append(Nos * num)
        # 后面插入5个0
        oversampled.extend([0] * behind)

    if graph == True:
        plt.stem(oversampled[:200])

    return oversampled


def raised_cosine_filter(data,Nos= 10,Rb = 1e9,graph = False):
    """
    生成升余弦滤波器系数
    :param alpha: 滚降因子(0-1)
    :param span: 滤波器符号长度
    :param sps: 每符号采样数
    :return: 滤波器系数
    """
    Nsample = len(data) # Length of the OOK_Imp array 样本长度
    fs = Nos * Rb #Sampling frequency (Hz) 采样频率
    df = fs / Nsample  # Frequency resolution
    # freq = np.linspace(-fs/2, fs/2, Nsample, endpoint=False) # Frequency vector (from -fs/2 to fs/2 with step size df)
    freq = np.arange(-fs / 2, fs / 2, df) # Frequency vector (from -fs/2 to fs/2 with step size df)
    omega = 2 * np.pi * freq # Angular frequency vector (omega = 2 * pi * freq)
    #生成矩形窗函数（频域）


    def myrect_fil(x, bound):
        return np.logical_and(x > -(bound / 2), x < (bound / 2)) 
    #频域乘是升余弦滚降函数，滚降系数为1
    NQ1_RC = (np.cos(omega / (4 * Rb)))**2 * myrect_fil(omega, 4 * np.pi * Rb)

    fftOOK_Imp = np.fft.fftshift(np.fft.fft(data))#对过采样信号进行傅里叶变换
    fftOOK_RC = fftOOK_Imp * NQ1_RC #对信号进行调制

    OOK_RC = np.real(np.fft.ifft(np.fft.ifftshift(fftOOK_RC))) #傅里叶逆变换

    #画图
    if graph == True:
        plt.plot(OOK_RC[:200])
    return OOK_RC

def diff_encoding(data):
    """差分编码实现
    
    参数:
        data - 原始输入数据序列(整数列表，每个元素应为0或1)
        
    返回:
        差分编码后的信号序列(列表)
    """
    encoded = []     # 存储编码结果
    prev_bit = 0     # 初始化前一个比特为0
    
    for bit in data:
        # 验证输入必须是二进制0/1
        if bit not in (0, 1):
            raise ValueError("输入数据必须为0或1的二进制序列")
        
        # 差分编码核心逻辑
        encoded_bit = prev_bit ^ bit
        encoded.append(encoded_bit)
        
        # 更新前一个比特
        prev_bit = encoded_bit
    
    return encoded


def prs_modulation(data,graph = False):
    '''
    进行prs调制
    '''
    
    o_data = np.pad(data, (0, 1), 'constant')
    d_data = np.pad(data, (1, 0), 'constant')  # 前面加1个0，后面不加


    prs_data = o_data + d_data

    # prs_signal = raised_cosine_filter(oversampling(prs_data,graph=False), Nos = Nos, Rb= 1e9 ,graph=False)
    
    if graph ==True:
        plt.figure(3)
        plt.plot(prs_data[:200])
        # plt.figure(6)
        # plt.stem(oversampling(o_data,graph=False)[:200])
        # plt.figure(7)
        # plt.stem(oversampling(d_data,graph=False)[:200])
    return prs_data


def AWGNGen(signal,EbN0_dB = 5,M = 2,Nos = 10,Rb = 1e9,graph = False, Actual_energy = True,noise_power_comparison = False):  # Function to make AWGN
    siglen = len(signal)
    ndata = siglen/Nos
    # t = np.arange(siglen) * ts
    fs = Nos * Rb #Sampling frequency (Hz) 采样频率
    ts = 1/fs

    if Actual_energy:
        Es = np.sum(signal ** 2) * ts  # Energy of the signal
        Eb = Es / ndata  # Energy per bit
    else:
        #这里没写理论能量
        Es = np.sum(signal ** 2) * ts  # Energy of the signal
        Eb = Es / ndata  # Energy per bit

    EbN0_linear = 10**(EbN0_dB/10)  # dB to linear
    N0 = Eb / EbN0_linear  # Power Spectrum Density
    N = N0 * Rb  # Noise power
    sigma = np.sqrt(N / 2)  # Standard deviation of noise
    noise = sigma * np.random.randn(siglen)  # Generate noise

    if noise_power_comparison:
        noise_power = np.mean(noise**2)  # 应≈N/2
        print(f"理论噪声功率: {N/2}, 实际噪声功率: {noise_power}")

    if graph :
        plt.plot(noise[:200])
    return noise


def awgn_channel(data,EbN0_db = 5,M = 2,Nos = 10,Rb = 1e9,graph = False):
    noisy_signal = AWGNGen(data,EbN0_db,M = M,Nos = Nos,Rb = Rb,graph = False) + data
    if graph == True :
        plt.plot(noisy_signal[:200])
        plt.plot(data[:200])
    return noisy_signal



def signal_detection(noisy_data,detection_points = [2,7],graph = False, Nos = 10):
    data = np.asarray(noisy_data)
    
    # 计算总符号数
    n_symbols = len(data) // Nos
    
    # 生成所有检测点的全局索引
    indices = np.array(detection_points)[:, None] + Nos * np.arange(n_symbols)
    indices = indices.ravel() 

    result = data[indices].reshape(1, -1)

    if graph :
        plt.figure(figsize=(6, 5))
        
        # 绘制原始信号
        plot_end = min(200, len(data))
        plt.plot(data[:plot_end], 'b-', alpha=0.7, label='Original Signal')
        
        # 标注检测点（在绘图范围内的）
        plot_indices = indices[indices < plot_end]
        plt.scatter(plot_indices, data[plot_indices], 
                    c='red', s=60, marker='o', 
                    label=f'Detection Points {detection_points}')
        
        # 添加符号周期分隔线
        for i in range(0, plot_end, Nos):
            plt.axvline(x=i, color='gray', linestyle=':', alpha=0.3)
        
        # 图表装饰
        plt.title(f'Signal Detection (First {plot_end} Samples)')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    return result


def signal_detection(noisy_data,detection_points = [2,7],graph = False,Nos = 10):
    """
    从信号中按固定步长 Nos 提取检测点样本。
    
    参数:
        noisy_data: 输入信号（1D 数组或列表）
        detection_points: 每个 Nos 块内的相对检测点（如 [0,1] 或 [3,6]）
        Nos: 外层步长（每 Nos 个样本为一个块）
        graph: 是否绘图（暂未实现）
    
    返回:
        提取的样本数组（形状: (1, num_samples)）
    """
    data = np.asarray(noisy_data)
    n_blocks = len(data) // Nos  # 计算完整块的数量
    
    # 生成全局索引: 每个块起始位置 + 检测点偏移
    block_starts = Nos * np.arange(n_blocks)  # [0, Nos, 2*Nos, ...]
    indices = block_starts[:, None] + detection_points  # 广播相加
    indices = indices.ravel()
    
    # 过滤越界索引
    indices = indices[indices < len(data)]
    
    # 提取数据并 reshape
    result = data[indices].reshape(1, -1)

    if graph :
        plt.figure(figsize=(6, 5))
        
        # 绘制原始信号
        plot_end = min(200, len(data))
        plt.plot(data[:plot_end], 'b-', alpha=0.7, label='Original Signal')
        
        # 标注检测点（在绘图范围内的）
        plot_indices = indices[indices < plot_end]
        plt.scatter(plot_indices, data[plot_indices], 
                    c='red', s=60, marker='o', 
                    label=f'Detection Points {detection_points}')
        
        # 添加符号周期分隔线
        for i in range(0, plot_end, Nos):
            plt.axvline(x=i, color='gray', linestyle=':', alpha=0.3)
        
        # 图表装饰
        plt.title(f'Signal Detection (First {plot_end} Samples)')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    return result


if __name__ =='__main__':
    # torch.manual_seed(0)
    # np.random.seed(0)
    # prs_signal = prs_modulation(info_data,Nos = 10,graph=False)
    # print(prs_signal.shape)
    # over = oversampling(info_data,graph=False)
    # print("over")
    signal = np.arange(1, 1001)

    res = signal_detection(signal, detection_points = [2,7],graph = True,Nos = 10)
    print(res[:200])
