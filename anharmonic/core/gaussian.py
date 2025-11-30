"""
高斯消元模块

用于求解线性方程组和确定独立基向量
"""

import numpy as np
from numpy.typing import NDArray
from numba import jit

# 数值精度阈值
EPSILON = 1e-10


@jit(nopython=True)
def gaussian_elimination(
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    特化的高斯消元法 (JIT 编译)
    
    用于确定线性方程组的独立变量和相关变量，
    并构建从独立变量到相关变量的变换矩阵。
    
    Args:
        matrix: 系数矩阵 (m, n)
        
    Returns:
        transformation_matrix: 变换矩阵 (n, n_independent)
        independent_indices: 独立变量的列索引
    """
    # 确保数组连续且不修改原数组
    matrix = np.ascontiguousarray(matrix)
    num_rows, num_cols = matrix.shape
    
    # 存储相关和独立变量的索引
    dependent_indices = np.empty(num_cols, dtype=np.int64)
    independent_indices = np.empty(num_cols, dtype=np.int64)
    transformation_matrix = np.zeros((num_cols, num_cols), dtype=np.float64)
    
    current_row = 0
    num_dependent = 0
    num_independent = 0
    
    for col in range(min(num_rows, num_cols)):
        # 将当前列中绝对值小于阈值的元素置零
        column = matrix[:, col]
        column[np.abs(column) < EPSILON] = 0.0
        
        # 选取主元：寻找绝对值最大的行并交换
        if current_row < num_rows:
            for row in range(current_row + 1, num_rows):
                if abs(matrix[row, col]) - abs(matrix[current_row, col]) > EPSILON:
                    # 交换行
                    temp = matrix[current_row, col:num_cols].copy()
                    matrix[current_row, col:num_cols] = matrix[row, col:num_cols]
                    matrix[row, col:num_cols] = temp
        
        if current_row < num_rows and abs(matrix[current_row, col]) > EPSILON:
            # 当前列是相关变量（主元列）
            dependent_indices[num_dependent] = col
            num_dependent += 1
            
            # 归一化主元行
            pivot = matrix[current_row, col]
            if num_cols - 1 > col:
                matrix[current_row, col + 1:num_cols] /= pivot
            matrix[current_row, col] = 1.0
            
            # 消去其他行在当前列的元素
            for row in range(num_rows):
                if row == current_row:
                    continue
                if num_cols - 1 > col:
                    matrix[row, col + 1:num_cols] -= (
                        matrix[row, col] * matrix[current_row, col + 1:num_cols] 
                        / matrix[current_row, col]
                    )
                matrix[row, col] = 0.0
            
            if current_row < num_rows - 1:
                current_row += 1
        else:
            # 当前列是独立变量
            independent_indices[num_independent] = col
            num_independent += 1
    
    # 构建变换矩阵
    for j in range(num_independent):
        for i in range(num_dependent):
            transformation_matrix[dependent_indices[i], j] = -matrix[i, independent_indices[j]]
        transformation_matrix[independent_indices[j], j] = 1.0
    
    return transformation_matrix, independent_indices[:num_independent]
