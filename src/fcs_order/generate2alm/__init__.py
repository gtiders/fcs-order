"""处理DFTSETS文件，用于alamode软件的数据准备。"""

import click
from ase.io import read

# 物理常数：转换因子
EV_TO_RYD = 1 / 13.60569253  # 1 eV = 1/13.60569253 Rydberg
ANGSTROM_TO_BOHR = 1 / 0.529177210903  # 1 Å = 1/0.529177210903 bohr
FORCE_CONV = EV_TO_RYD / ANGSTROM_TO_BOHR  # eV/Å -> Ryd/bohr


def calculate_properties(atoms, base_atom):
    """
    计算从基准结构到当前结构的位移和力

    参数:
        atoms: 当前结构 (ASE Atoms对象)
        base_atom: 基准结构 (ASE Atoms对象)

    返回:
        tuple: (displacements, forces, potential_energy)
            displacements: 原子位移数组 (bohr单位)
            forces: 原子力数组 (Ryd/bohr单位)
            potential_energy: 势能 (Rydberg单位)
    """

    displacements = (
        atoms.get_positions() * ANGSTROM_TO_BOHR
        - base_atom.get_positions() * ANGSTROM_TO_BOHR
    )
    forces = atoms.get_forces() * FORCE_CONV
    potential_energy = atoms.get_potential_energy() * EV_TO_RYD
    return displacements, forces, potential_energy
