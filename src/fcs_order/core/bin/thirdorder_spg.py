import numpy as np
import spglib


class SpglibDataset:
    """Python版本的SpglibDataset结构体，模拟C API"""

    def __init__(self):
        self.spacegroup_number = 0
        self.hall_number = 0
        self.international_symbol = ""
        self.hall_symbol = ""
        self.transformation_matrix = np.zeros((3, 3))
        self.origin_shift = np.zeros(3)
        self.n_operations = 0
        self.rotations = None
        self.translations = None
        self.n_atoms = 0
        self.wyckoffs = None
        self.equivalent_atoms = None


def spg_get_dataset(lattice, positions, types, num_atom, symprec):
    """
    模拟spg_get_dataset C函数
    参数格式与C API完全一致
    """
    # 转换输入格式为spglib期望的格式
    lattice_array = np.array(lattice).reshape(3, 3).T  # spglib期望列向量格式
    positions_array = np.array(positions).reshape(-1, 3)
    types_array = np.array(types)

    # 调用Python版本的spglib
    dataset_dict = spglib.get_symmetry_dataset(
        (lattice_array, positions_array, types_array), symprec=symprec
    )

    if dataset_dict is None:
        return None

    # 创建SpglibDataset对象并填充数据
    dataset = SpglibDataset()

    dataset.spacegroup_number = dataset_dict["number"]
    dataset.hall_number = dataset_dict.get("hall_number", 0)
    dataset.international_symbol = dataset_dict["international"]
    dataset.hall_symbol = dataset_dict.get("hall_symbol", "")

    # 转换矩阵和原点偏移
    dataset.transformation_matrix = np.array(dataset_dict["transformation_matrix"])[
        :3, :3
    ]
    dataset.origin_shift = np.array(dataset_dict["origin_shift"])[:3]

    # 对称操作
    dataset.n_operations = len(dataset_dict["rotations"])
    dataset.rotations = np.array(dataset_dict["rotations"], dtype=np.int32)
    dataset.translations = np.array(dataset_dict["translations"], dtype=np.float64)

    # 原子信息
    dataset.n_atoms = len(dataset_dict["wyckoffs"])
    dataset.wyckoffs = np.array(dataset_dict["wyckoffs"], dtype=np.int32)
    dataset.equivalent_atoms = np.array(
        dataset_dict["equivalent_atoms"], dtype=np.int32
    )

    return dataset


def spg_free_dataset(dataset):
    """
    模拟spg_free_dataset C函数
    在Python中不需要手动释放内存，但提供接口以保持兼容性
    """
    pass
