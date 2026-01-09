# MLFCS (Machine Learning Force Constant Suite)

![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

MLFCS 是一个现代化的非谐力常数（Anharmonic Force Constants）计算工具套件，旨在为高通量材料计算提供高效、易用的解决方案。

本项目基于经典的 `thirdorder.py` 和 `fourthorder.py` 进行了深度重构与优化。

## ✨ 核心特性

*   **纯 Python 实现**：彻底移除了对 `syplib` C 扩展库的依赖，解决了繁琐的编译和依赖问题。
*   **极致性能**：
    *   🚀 **速度提升 5 倍**：经过算法优化，计算速度大幅提升。
    *   💾 **内存占用仅 1%**：内存管理极大优化，轻松处理大规模超胞体系。
    *   📦 **极简安装**：支持标准的 `pip` 安装流程，开箱即用。
*   **功能完备**：支持三阶（Third-order）和四阶（Fourth-order）力常数的生成（Sow）与提取（Reap）。
*   **多格式支持**：兼容 **VASP** 和 **XYZ/ExtXYZ** 格式，方便与多种计算代码（如 ASE calculators）集成。

## 🛠️ 安装

您可以直接通过 pip 安装本项目：

```bash
git clone https://github.com/gtiders/mlfcs.git
cd mlfcs
pip install .
```

## 📖 使用指南

本套件包含两个主要命令：`thirdorder` 和 `fourthorder`。每个命令都包含 `sow`（生成位移）和 `reap`（收集力并计算力常数）两个子命令。

### 三阶力常数计算 (Thirdorder)

#### 1. 生成位移结构 (Sow)

生成用于计算三阶力常数的超胞位移结构。

```bash
# 基本用法：生成 4x4x4 超胞，截断半径第 3 近邻 (负数表示近邻层数)
thirdorder sow 4 4 4 -3

# 指定截断半径为 5.0 nm (正数表示距离)
thirdorder sow 4 4 4 5.0

# 输出为 xyz 格式 (推荐用于机器学习势)
thirdorder sow 4 4 4 -3 --format xyz

# 自定义参数: 位移步长 0.001, 对称性精度 1e-4
thirdorder sow 4 4 4 -3 --hstep 0.001 --symprec 1e-4
```

#### 2. 计算力 (外部步骤)

如果您使用了 `xyz` 格式，您需要使用自己的计算器（如 VASP, LAMMPS, 或者机器学习势）计算 `3RD.displacements.xyz` 中所有结构的力。

**关键点**：确保计算后的文件包含力信息，并且结构顺序保持不变（或者包含 `config_id` 属性）。

#### 3. 收集力常数 (Reap)

使用 `reap` 命令从计算好的文件中提取力常数。

```bash
# 基本用法：从 VASP xml/OUTCAR 文件中提取
thirdorder reap 4 4 4 -3 --forces vasprun.xml.*

# 进阶用法：从计算好的 XYZ 文件中提取 (例如：calculated_forces.xyz)
# 注意：如果 sow 时使用了非默认的 hstep，reap 时必须指定相同的 hstep
thirdorder reap 4 4 4 -3 --forces calculated_forces.xyz --hstep 0.001
```

结果将输出到 `FORCE_CONSTANTS_3RD` 文件。

### 四阶力常数计算 (Fourthorder)

操作流程与三阶类似。

#### 1. 生成位移结构 (Sow)

```bash
# 生成 3x3x3 超胞，第 2 近邻截断
fourthorder sow 3 3 3 -2
```

这将生成 `4TH.POSCAR.*` 文件。

#### 2. 计算力常数 (Reap)

```bash
fourthorder reap 3 3 3 -2 --forces vasprun.xml.*
```

结果将输出到 `FORCE_CONSTANTS_4TH` 文件。

## 🐍 Python API 调用 (进阶用法)

除了命令行工具，您也可以直接在 Python 脚本中调用核心类，这对于集成 ASE 计算器（如 NEP, GAP, MACE, DP 等）非常方便，无需中间文件读写。

### 基础示例

```python
from mlfcs.thirdorder import ThirdOrderRun
# 假设您使用 calorine 的 CPUNEP 计算器，也可以是任何 ASE Calculator
from calorine.calculators import CPUNEP

# 初始化运行器
# 参数: na=4, nb=4, nc=4, cutoff=-3 (第3近邻)
runner = ThirdOrderRun(4, 4, 4, -3)

# 定义 ASE 计算器
calc = CPUNEP("nep.txt")

# 直接运行计算，无需手动处理文件 I/O
runner.run_calculator(calc)
```

### 参数覆盖 (H & Symprec)

您可以在初始化时自定义位移步长 (`h`) 和对称性精度 (`symprec`)：

```python
# h: 位移步长 (默认通常为 0.001 或类似值，具体取决于阶数)
# symprec: 对称性判断精度 (默认 1e-5)
runner = ThirdOrderRun(4, 4, 4, -3, h=0.001, symprec=1e-4)
```

### 自洽谐波计算 (SSCHA)

您可以使用 `MLPSSCHA` 类结合任意 ASE 计算器（如 NEP）进行 SSCHA 计算。

```python
from mlfcs.sscha import MLPSSCHA
from calorine.calculators import CPUNEP

# 初始化计算器
calc = CPUNEP("nep.txt")

# 设置 SSCHA 运行参数
sscha = MLPSSCHA(
    unitcell="./POSCAR",         # 原胞文件路径
    supercell_matrix=[3, 3, 3],  # 超胞扩倍矩阵
    calculator=calc,             # ASE 计算器
    temperature=300,             # 温度 (K)
    number_of_snapshots=1000,    # 每次迭代生成的结构数
    max_iterations=20,           # 最大迭代次数
    avg_n_last_steps=5,          # 取最后 5 步结果平均作为最终输出
    fc_output="FORCE_CONSTANTS"  # 输出文件名
)

# 运行计算
sscha.run()
```

### 最佳实践：防止计算器缓存问题

如果您选择手动遍历结构进行计算（而不是使用 `runner.run_calculator`），请务必注意 ASE 计算器的缓存机制。为了防止 `write` 操作意外触发重算或写入旧数据，建议使用 `SinglePointCalculator` "冻结" 结果。

```python
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

# ... 假设在循环中 ...
atoms.calc = calc  # 挂载您的主计算器 (如 NEP, VASP 等)
forces = atoms.get_forces()
energy = atoms.get_potential_energy()

# 【关键步骤】卸载主计算器，使用 SinglePointCalculator 存储静态结果
# 这样可以安全地写入文件，避免重触发 calc 计算，也避免多帧数据混淆
atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)

# 现在可以安全写入
write("forces.xyz", atoms, format="extxyz", append=True)
```

## 🙏 致谢

本项目的开发离不开开源社区的贡献，特别感谢以下先驱项目：

*   **[ShengBTE / thirdorder.py](https://www.shengbte.org/announcements/thirdorderpyv110released)**: 感谢 Wu Li 等人开发的原始 `thirdorder.py`，为非谐声子计算奠定了基础。
*   **[Fourthorder](https://github.com/FourPhonon/Fourthorder)**: 感谢 Han, Zherui 等人开发的四阶力常数计算代码。

我们在这些优秀工作的基础上，重点改进了软件工程架构、安装体验以及运行效率，希望能为社区提供更好用的工具。

## 📄 许可证

本项目遵循 GNU General Public License v3.0 (GPLv3) 许可证。
