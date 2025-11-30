"""
四阶力常数 JIT 编译内核函数

集中管理四阶计算专用的 numba JIT 编译函数
"""

from numba import jit
from numba.typed import List
import numpy as np
from numpy.typing import NDArray


# ============================================================================
# 四元组操作函数
# ============================================================================

@jit(nopython=True)
def jit_quartet_in_list(
    quartet: NDArray[np.intc],
    quartet_list: NDArray[np.intc],
    list_length: int,
) -> bool:
    """
    检查四元组是否在列表中
    
    Args:
        quartet: 要检查的四元组 (4,)
        quartet_list: 四元组列表 (4, N)
        list_length: 列表中有效四元组的数量
        
    Returns:
        如果找到则返回 True
    """
    for i in range(list_length):
        if (
            quartet[0] == quartet_list[0, i]
            and quartet[1] == quartet_list[1, i]
            and quartet[2] == quartet_list[2, i]
            and quartet[3] == quartet_list[3, i]
        ):
            return True
    return False


@jit(nopython=True)
def jit_quartets_equal(
    quartet1: NDArray[np.intc],
    quartet2: NDArray[np.intc],
) -> bool:
    """
    比较两个四元组是否相等
    
    Args:
        quartet1: 第一个四元组 (4,)
        quartet2: 第二个四元组 (4,)
        
    Returns:
        如果相等则返回 True
    """
    for i in range(4):
        if quartet1[i] != quartet2[i]:
            return False
    return True


# ============================================================================
# 四阶变换数组构建函数
# ============================================================================

@jit(nopython=True)
def jit_build_transformation_array4(
    transformation: NDArray[np.float64],
    transformation_auxiliary: NDArray[np.float64],
    equivalent_count: NDArray[np.intc],
    independent_basis_count: NDArray[np.intc],
    num_quartet_classes: int,
    output_array: NDArray[np.float64],
) -> None:
    """
    构建四阶力常数的变换数组
    
    计算:
        output[k, l, j, i] = sum_{aux} transformation[k, aux, j, i] 
                                     * transformation_auxiliary[aux, l, i]
    
    Args:
        transformation: 变换矩阵 (81, 81, max_equiv, nlist)
        transformation_auxiliary: 辅助变换矩阵 (81, 81, nlist)
        equivalent_count: 每个四元组类的等价数量 (nlist,)
        independent_basis_count: 每个四元组类的独立基数量 (nlist,)
        num_quartet_classes: 四元组类的数量
        output_array: 输出数组 (81, 81, max_equiv, nlist)
    """
    for quartet_idx in range(num_quartet_classes):
        num_equiv = equivalent_count[quartet_idx]
        num_independent = independent_basis_count[quartet_idx]
        
        for equiv_idx in range(num_equiv):
            for basis_prime in range(81):
                for independent_idx in range(num_independent):
                    value = 0.0
                    for aux_idx in range(81):
                        value += (
                            transformation[basis_prime, aux_idx, equiv_idx, quartet_idx]
                            * transformation_auxiliary[aux_idx, independent_idx, quartet_idx]
                        )
                    if value != 0.0 and abs(value) < 1e-15:
                        value = 0.0
                    output_array[basis_prime, independent_idx, equiv_idx, quartet_idx] = value


# ============================================================================
# 四阶 IFC 重构函数
# ============================================================================

@jit(nopython=True)
def jit_build_sparse_ifc4_indices(
    quartet_class_indices: NDArray[np.intc],
    equivalent_indices: NDArray[np.intc],
    accumulated_independent: NDArray[np.intc],
    num_primitive_atoms: int,
    num_supercell_atoms: int,
    num_quartet_classes: int,
    transformation_array: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """
    构建稀疏四阶力常数矩阵的 COO 格式索引和值
    
    Args:
        quartet_class_indices: 四元组类索引 (natoms, ntot, ntot, ntot)
        equivalent_indices: 等价索引 (natoms, ntot, ntot, ntot)
        accumulated_independent: 累积独立基数量 (nlist + 1,)
        num_primitive_atoms: 原胞原子数
        num_supercell_atoms: 超胞原子数
        num_quartet_classes: 四元组类数量
        transformation_array: 变换数组 (81, 81, max_equiv, nlist)
        
    Returns:
        row_indices: 行索引
        col_indices: 列索引
        values: 非零值
    """
    # 第一遍：计算总元素数
    total_count = 0
    for atom_i in range(num_primitive_atoms):
        for atom_j in range(num_supercell_atoms):
            for atom_k in range(num_supercell_atoms):
                for atom_m in range(num_supercell_atoms):
                    class_idx = quartet_class_indices[atom_i, atom_j, atom_k, atom_m]
                    if class_idx >= 0:
                        total_count += (
                            accumulated_independent[class_idx + 1]
                            - accumulated_independent[class_idx]
                        ) * 81
    
    # 分配数组
    row_indices = np.empty(total_count, dtype=np.int64)
    col_indices = np.empty(total_count, dtype=np.int64)
    values = np.empty(total_count, dtype=np.float64)
    
    # 第二遍：填充数组
    count = 0
    col_index = 0
    
    for atom_i in range(num_primitive_atoms):
        for atom_j in range(num_supercell_atoms):
            for atom_k in range(num_supercell_atoms):
                quad_basis_index = 0
                for coord_l in range(3):
                    for coord_m in range(3):
                        for coord_n in range(3):
                            for coord_o in range(3):
                                for atom_m in range(num_supercell_atoms):
                                    class_idx = quartet_class_indices[atom_i, atom_j, atom_k, atom_m]
                                    if class_idx >= 0:
                                        start = accumulated_independent[class_idx]
                                        end = accumulated_independent[class_idx + 1]
                                        equiv_idx = equivalent_indices[atom_i, atom_j, atom_k, atom_m]
                                        for s in range(start, end):
                                            independent_idx = s - start
                                            row_indices[count] = s
                                            col_indices[count] = col_index
                                            values[count] = transformation_array[
                                                quad_basis_index,
                                                independent_idx,
                                                equiv_idx,
                                                class_idx,
                                            ]
                                            count += 1
                                quad_basis_index += 1
                                col_index += 1
    
    return row_indices[:count], col_indices[:count], values[:count]


@jit(nopython=True)
def jit_build_full_ifc4_coordinates(
    num_quartet_classes: int,
    equivalent_count: NDArray[np.intc],
    independent_basis_count: NDArray[np.intc],
    equivalent_quartets: NDArray[np.intc],
    accumulated_independent: NDArray[np.intc],
    transformation_array: NDArray[np.float64],
    force_constant_values: NDArray[np.float64],
) -> tuple:
    """
    构建完整四阶力常数的坐标和值列表
    
    Args:
        num_quartet_classes: 四元组类数量
        equivalent_count: 等价数量数组 (nlist,)
        independent_basis_count: 独立基数量数组 (nlist,)
        equivalent_quartets: 等价四元组数组 (4, max_equiv, nlist)
        accumulated_independent: 累积独立基数量 (nlist + 1,)
        transformation_array: 变换数组 (81, 81, max_equiv, nlist)
        force_constant_values: 力常数值数组 (total_independent,)
        
    Returns:
        八个坐标列表和一个值列表
    """
    coord_0 = List()  # coord_l
    coord_1 = List()  # coord_m
    coord_2 = List()  # coord_n
    coord_3 = List()  # coord_o
    coord_4 = List()  # atom_i
    coord_5 = List()  # atom_j
    coord_6 = List()  # atom_k
    coord_7 = List()  # atom_l
    values = List()
    
    for class_idx in range(num_quartet_classes):
        num_equiv = equivalent_count[class_idx]
        num_independent = independent_basis_count[class_idx]
        
        for equiv_idx in range(num_equiv):
            atom_i = equivalent_quartets[0, equiv_idx, class_idx]
            atom_j = equivalent_quartets[1, equiv_idx, class_idx]
            atom_k = equivalent_quartets[2, equiv_idx, class_idx]
            atom_m = equivalent_quartets[3, equiv_idx, class_idx]
            
            for coord_l in range(3):
                for coord_m in range(3):
                    for coord_n in range(3):
                        for coord_o in range(3):
                            quad_basis_index = ((coord_l * 3 + coord_m) * 3 + coord_n) * 3 + coord_o
                            
                            for independent_idx in range(num_independent):
                                value = (
                                    transformation_array[quad_basis_index, independent_idx, equiv_idx, class_idx]
                                    * force_constant_values[accumulated_independent[class_idx] + independent_idx]
                                )
                                
                                if value == 0.0:
                                    continue
                                
                                coord_0.append(coord_l)
                                coord_1.append(coord_m)
                                coord_2.append(coord_n)
                                coord_3.append(coord_o)
                                coord_4.append(atom_i)
                                coord_5.append(atom_j)
                                coord_6.append(atom_k)
                                coord_7.append(atom_m)
                                values.append(value)
    
    return coord_0, coord_1, coord_2, coord_3, coord_4, coord_5, coord_6, coord_7, values


# ============================================================================
# 向后兼容的别名
# ============================================================================

_quartet_in_list = jit_quartet_in_list
_quartets_are_equal = jit_quartets_equal
_build_transformationarray4 = jit_build_transformation_array4
_build_ijv_fourthorder = jit_build_sparse_ifc4_indices
_build_ifc4_coords_vals_list = jit_build_full_ifc4_coordinates
