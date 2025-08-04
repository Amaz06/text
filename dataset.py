import numpy as np
from utils import truple_to_one_hot


def generate_dataset(clean_signal, noisy_signal,samples_per_symbol):
    # 将信号转换为张量
    padded = np.concatenate([[0], clean_signal, [0]])
    clean_window_size = 3 
    label_rows = len(padded) - clean_window_size + 1
    label_shape = (label_rows, clean_window_size)
    label_signal = np.lib.stride_tricks.as_strided(padded, label_shape, strides=padded.strides + padded.strides, subok=False, writeable=True)
    # print(label_signal[:10,:])
    label_signal = truple_to_one_hot(label_signal)
    # print(clean_signal[:10])
    # print(label_signal[:10,:])

    
    input_window = samples_per_symbol + (samples_per_symbol // 2) *2
    # input_window = samples_per_symbol * 3
    input_padded = np.concatenate([np.zeros((1,samples_per_symbol//2)), noisy_signal, np.zeros((1,samples_per_symbol//2))],axis=1)
    # input_padded = np.concatenate([np.zeros((1,2*(samples_per_symbol//2))), noisy_signal, np.zeros((1,2*(samples_per_symbol//2)))],axis=1)
    input_rows = label_rows
    input_shape = (input_rows, input_window)
    input_signal = np.lib.stride_tricks.as_strided(input_padded, input_shape, strides=(input_padded.strides[1]* samples_per_symbol, input_padded.strides[1]), subok=False, writeable=True)
    # print(noisy_signal[0,:10])
    # print(input_signal[:5,:])

    return input_signal, label_signal
