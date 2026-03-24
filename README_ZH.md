# MLFCS (Machine Learning Force Constant Suite)

![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

MLFCS 是一个现代化的非谐力常数（Anharmonic Force Constants）计算工具套件，旨在为高通量材料计算提供高效、易用的解决方案。

本项目基于经典的 `thirdorder.py` 和 `fourthorder.py` 进行了深度重构与优化。

## 🤖 AI 辅助文档

**初次使用？** 您可以：

*   📖 **将本 README 喂给 AI 助手**（如 ChatGPT、Claude、DeepSeek 等），快速了解用法并上手。
*   💻 **将整个代码库喂给 AI 助手**，深入理解实现细节和高级用法。

同时欢迎社区贡献：
*   🐛 **报告 Bug 或提出功能需求**：[提交 Issue](https://github.com/gtiders/mlfcs/issues)
*   🔧 **提交改进**：欢迎 [Pull Requests](https://github.com/gtiders/mlfcs/pulls)！
*   📧 **邮件联系**：gtiders@qq.com

## ⚠️ 版本说明

`main` 分支包含正在开发和实验性的功能（如 C++ `unordered_map` 优化）。生产环境请使用 [releases](https://github.com/gtiders/mlfcs/releases) 版本。

## 📋 输出格式

MLFCS 以原生格式输出力常数。**不提供 phono3py 格式的内置支持。** 如需 phono3py 兼容格式，可使用 [hiPhive](https://hiphive.materialsmodeling.org/) 进行格式转换。示例：

```python
from hiphive import ForceConstants

# 读取 MLFCS 输出并转换为 phono3py 格式
# 详见 hiphive 文档
```

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

### ⚠️ 古老系统注意事项 (CentOS 7 等)

在编译器版本较低的系统上（GCC < 9），NumPy 2.0+ 可能导致编译问题。安装前请修改 `pyproject.toml`：

```diff
- requires = ["setuptools>=80.0.0", "wheel", "cython>=3.0.0", "numpy>=2.0.0"]
+ requires = ["setuptools>=80.0.0", "wheel", "cython>=3.0.0", "numpy<2.0.0"]
```

然后执行安装：

```bash
pip install .
```

## 📖 使用指南

本套件包含两个主要命令：`thirdorder` 和 `fourthorder`。每个命令都包含 `sow`（生成位移）和 `reap`（收集力并计算力常数）两个子命令。

### 接口选择（非 VASP 输入必须显式指定）

结构读取使用显式 `--interface` 参数：

- 默认是 `--interface vasp`。
- ABACUS 的 `STRU` 文件请使用 `--interface abacus`。
- 若读取失败，MLFCS 会打印 phonopy 支持的接口列表（如 `abacus`、`vasp`、`qe`、`cp2k`、`aims`）。

### 最小完整流程（Sow -> 计算 -> Reap）

```bash
# 1) 生成位移结构
thirdorder sow 4 4 4 --cutoff -3 --format vasp

# 2) 对每个位移结构做外部力计算
# 推荐目录：按 sow 编号建子目录
# 例如 3RD_runs/0001/vasprun.xml, 3RD_runs/0002/vasprun.xml, ...

# 3) 收集三阶力常数（排序后的目录顺序 -> 位移编号顺序）
thirdorder reap 4 4 4 --cutoff -3 \
  --forces $(find ./3RD_runs -name "vasprun.xml" | sort -V) \
  --forces-interface vasp
```

### 三阶力常数计算 (Thirdorder)

#### 1. 生成位移结构 (Sow)

生成用于计算三阶力常数的超胞位移结构。

```bash
# 基本用法：生成 4x4x4 超胞，截断半径第 3 近邻 (负数表示近邻层数)
thirdorder sow 4 4 4 --cutoff -3

# ABACUS 输入 (STRU)
thirdorder sow 4 4 4 --cutoff -3 -i STRU --interface abacus

# QE 输入示例
thirdorder sow 4 4 4 --cutoff -3 -i qe.in --interface qe

# CP2K 输入示例
thirdorder sow 4 4 4 --cutoff -3 -i cp2k.inp --interface cp2k

# 指定截断半径为 5.0 nm (正数表示距离)
thirdorder sow 4 4 4 --cutoff 5.0

# 输出为 xyz 格式 (推荐用于机器学习势)
thirdorder sow 4 4 4 --cutoff -3 --format xyz

# 自定义参数: 位移步长 0.001, 对称性精度 1e-4
thirdorder sow 4 4 4 --cutoff -3 --hstep 0.001 --symprec 1e-4
```

#### 2. 计算力 (外部步骤)

请对 `sow` 生成的每个位移结构执行力计算：
- 若 `sow --format vasp`：对 `3RD.POSCAR.*` 逐个计算。
- 若 `sow --format xyz`：从 `3RD.displacements.xyz` 生成并计算（常用于 Python 工作流）。

**关键点**：CLI `reap` 使用 phonopy 接口解析输出文件，务必保证文件命名/顺序可确定。

#### 3. 收集力常数 (Reap)

使用 `reap` 命令从计算好的文件中提取力常数。

```bash
# 基本用法：从 VASP xml/OUTCAR 文件中提取
thirdorder reap 4 4 4 --cutoff -3 --forces vasprun.xml.* --forces-interface vasp

# ABACUS 输出日志
thirdorder reap 4 4 4 --cutoff -3 --forces running_scf.log.* --forces-interface abacus
```

结果将输出到 `FORCE_CONSTANTS_3RD` 文件。

说明：
- CLI 的 `reap` 现在通过 phonopy 接口解析力文件，不再接受 `xyz/extxyz`。
- 若要使用 `xyz` 力轨迹，请使用 Python 库函数工作流。

##### Reap 多文件顺序规则

`reap` 会将“排序后的文件顺序”映射到位移编号（`1..N`），所以请确保输入顺序是确定的。
推荐按 `sow` 输出编号组织目录（如 `0001/`、`0002/` 等）。

```bash
# VASP: 简单通配符（文件名有补零或天然有序时可用）
thirdorder reap 4 4 4 --cutoff -3 --forces vasprun.xml.* --forces-interface vasp

# ABACUS: 显式 find + 版本排序
thirdorder reap 4 4 4 --cutoff -3 \
  --forces $(find ./abacus_runs -name "running_scf.log.*" | sort -V) \
  --forces-interface abacus

# QE: 收集 pw.x 输出
thirdorder reap 4 4 4 --cutoff -3 \
  --forces $(find ./qe_runs -name "pw.out.*" | sort -V) \
  --forces-interface qe

# CP2K: 收集输出日志
thirdorder reap 4 4 4 --cutoff -3 \
  --forces $(find ./cp2k_runs -name "*.out" | sort -V) \
  --forces-interface cp2k
```

如果不确定顺序，先打印确认：
```bash
find ./abacus_runs -name "running_scf.log.*" | sort -V
```

### 四阶力常数计算 (Fourthorder)

操作流程与三阶类似。

#### 1. 生成位移结构 (Sow)

```bash
# 生成 3x3x3 超胞，第 2 近邻截断
fourthorder sow 3 3 3 --cutoff -2

# ABACUS 输入 (STRU)
fourthorder sow 3 3 3 --cutoff -2 -i STRU --interface abacus

# QE 输入示例
fourthorder sow 3 3 3 --cutoff -2 -i qe.in --interface qe

# CP2K 输入示例
fourthorder sow 3 3 3 --cutoff -2 -i cp2k.inp --interface cp2k
```

这将生成 `4TH.POSCAR.*` 文件。

#### 2. 计算力常数 (Reap)

```bash
fourthorder reap 3 3 3 --cutoff -2 --forces vasprun.xml.* --forces-interface vasp

# ABACUS 输出日志
fourthorder reap 3 3 3 --cutoff -2 --forces running_scf.log.* --forces-interface abacus

# QE 输出（顺序规则同 thirdorder）
fourthorder reap 3 3 3 --cutoff -2 \
  --forces $(find ./qe_runs -name "pw.out.*" | sort -V) \
  --forces-interface qe
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

### 谐波声子计算 (MLPHONON)

您可以使用 `MLPHONON` 类结合任意 ASE 计算器计算谐波力常数。

```python
from mlfcs.phonon import MLPHONON
from ase.io import read
from calorine.calculators import CPUNEP

# 读取结构
structure = read("POSCAR")

# 初始化计算器
calc = CPUNEP("nep.txt")

# 设置声子计算
phonon = MLPHONON(
    structure=structure,
    calculator=calc,
    supercell_matrix=[2, 2, 2],  # 超胞扩倍矩阵
    kwargs_generate_displacements={"distance": 0.01}  # 可选参数
)

# 运行计算
phonon.run()

# 导出力常数到文件
phonon.write("FORCE_CONSTANTS")

# 访问 Phonopy 对象进行后续分析
phonon.phonopy.run_mesh([20, 20, 20])
phonon.phonopy.run_total_dos()
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
