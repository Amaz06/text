import numpy as np


def bits_to_int(bits):
    return bits[0]*4 + bits[1]*2 + bits[2]


def int_to_bits(val):
    return [ (val >> 2) & 1, (val >> 1) & 1, val & 1 ]


def binary_to_one_hot(binary_matrix):
    # 将每一行的二进制数转换为十进制索引
    indices = binary_matrix[:, 0] * 9 + binary_matrix[:, 1] * 3 + binary_matrix[:, 2] * 1
    
    # 生成独热编码矩阵
    one_hot_encoded = np.eye(27)[indices]  # 8 类独热编码
    
    return one_hot_encoded



# def indices_to_ternary(indices, num_digits=3):
#     """将类别索引转换为 3 位三进制矩阵
    
#     Args:
#         indices: 类别索引数组（0-26）
#         num_digits: 位数（默认为3，因为3位三进制可以表示0-26）
    
#     Returns:
#         ndarray: 形状为 (n_samples, 3) 的三进制矩阵
#     """
#     ternary_matrix = np.zeros((len(indices), num_digits), dtype=int)
    
#     for i, index in enumerate(indices):
#         for j in range(num_digits):
#             ternary_matrix[i, num_digits-1-j] = index % 3
#             index = index // 3
    
#     return ternary_matrix


def indices_to_ternary(indices, num_digits=3):
    """将预测的类别索引 (0-14) 解码为对应的三元组
    
    Args:
        indices: 类别索引数组（0-14）
        num_digits: 位数（固定为3，表示三元组长度）
    
    Returns:
        ndarray: 形状为 (n_samples, 3) 的三进制矩阵
    """
    # 定义有效状态的反向映射（索引 -> 三元组）
    index_to_ternary = {
        0: [0, 0, 0],
        1: [0, 0, 1],
        2: [0, 1, 1],
        3: [0, 1, 2],
        4: [1, 0, 0],
        5: [1, 0, 1],
        6: [1, 1, 0],
        7: [1, 1, 1],
        8: [1, 1, 2],
        9: [1, 2, 1],
        10: [1, 2, 2],
        11: [2, 1, 0],
        12: [2, 1, 1],
        13: [2, 2, 1],
        14: [2, 2, 2]
    }
    
    ternary_matrix = np.zeros((len(indices), num_digits), dtype=int)
    
    for i, index in enumerate(indices):
        if index in index_to_ternary:
            ternary_matrix[i] = index_to_ternary[index]
        else:
            raise ValueError(f"无效的类别索引 {index}，必须在 0-14 范围内")
    
    return ternary_matrix




def filter_valid_states(triplets):
    """
    将27种可能的三元组映射到15种有效状态，无效状态设为-1或其他标识
    """
    # 定义允许的状态组合（示例，需替换为你的实际规则）
    valid_states = {
        (0, 0, 0): 0,
        (0, 0, 1): 1,
        (0, 1, 1): 2,
        (0, 1, 2): 3,
        (1, 0, 0): 4,
        (1, 0, 1): 5,
        (1, 1, 0): 6,
        (1, 1, 1): 7,
        (1, 1, 2): 8,
        (1, 2, 1): 9,
        (1, 2, 2): 10,
        (2, 1, 0): 11,
        (2, 1, 1): 12,
        (2, 2, 1): 13,
        (2, 2, 2): 14  # 最后一个有效状态
    }


def truple_to_one_hot(binary_matrix):
    """
    将二进制三元组映射到15种有效状态,无效状态返回全零向量或报错。

    参数:
        binary_matrix: 形状为 (N, 3) 的二进制矩阵，每行是一个三元组 (a, b, c)

    返回:
        one_hot_encoded: 形状为 (N, 15) 的独热编码矩阵
    """
    # 定义有效状态映射规则
    valid_states = {
        (0, 0, 0): 0,
        (0, 0, 1): 1,
        (0, 1, 1): 2,
        (0, 1, 2): 3,
        (1, 0, 0): 4,
        (1, 0, 1): 5,
        (1, 1, 0): 6,
        (1, 1, 1): 7,
        (1, 1, 2): 8,
        (1, 2, 1): 9,
        (1, 2, 2): 10,
        (2, 1, 0): 11,
        (2, 1, 1): 12,
        (2, 2, 1): 13,
        (2, 2, 2): 14
    }

    # 初始化输出矩阵（全零，表示无效状态）
    one_hot_encoded = np.zeros((binary_matrix.shape[0], 15))

    for i, triplet in enumerate(binary_matrix):
        # 将三元组转换为元组（因为 NumPy 数组不可哈希）
        triplet_tuple = tuple(triplet)
        
        # 检查是否为有效状态
        if triplet_tuple in valid_states:
            state_idx = valid_states[triplet_tuple]
            one_hot_encoded[i, state_idx] = 1  # 设置独热编码
        else:
            # 可选：无效状态处理（默认全零，或抛出警告）
            print(f"警告：无效的三元组 {triplet_tuple}，已设为全零")

    return one_hot_encoded