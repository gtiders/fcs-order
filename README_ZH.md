# MLFCS

<p align="center">
  <strong>Machine Learning Force Constant Suite</strong><br/>
  面向二阶/三阶/四阶力常数计算的实用工具套件（CLI + Python API）
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-GPLv3-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue.svg">
  <img alt="Version" src="https://img.shields.io/badge/version-2.0.0-0A7EA4.svg">
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="docs/quickstart.md">快速上手</a>
</p>

## 项目定位

MLFCS 是一个面向材料热学与非谐声子研究的力常数计算套件，当前围绕三条主线：

- `thirdorder` / `fourthorder` 命令行流程：`sow` + `reap`，适合传统文件工作流。
- `secondorder` Python API：`MLPHONON`、`MLPSSCHA`，适合直接调用 ASE 计算器。
- `hifinit` 单类接口：`HifinitRun`，先用有限差分直接计算目标阶数力常数，再投影到 hiPhive 参数空间；相比纯拟合法更容易控制阶间解耦，避免高阶/低阶参数相互“污染”。

目标是：在保证物理流程清晰的前提下，提高可脚本化能力、接口清晰度和工程可维护性。

## 为什么使用 MLFCS

- 一套代码覆盖 2nd / 3rd / 4th 阶力常数。
- 同时支持 CLI 文件流与 Python 直连计算流。
- 显式接口控制：结构读写与力解析均可通过 phonopy 接口指定。
- 输出兼容 phonopy / phono3py / ShengBTE 常见后处理路径。

## HIFINIT 设计说明

`HifinitRun` 的核心思路不是“直接把一组力数据拿去全局拟合”，而是：

1. 用有限差分在轨道原型上直接构造各阶力常数张量。
2. 再将结果映射到 hiPhive 的参数化空间，用于施加对称性与声学求和规则（ASR）等约束。

这种流程的主要价值：
- 各阶物理量来源更清晰：每一阶来自对应阶的有限差分构造，而不是由全局拟合共同分配。
- 降低阶间耦合误差：能更好避免纯拟合法里常见的“低阶/高阶互相吸收误差”问题。
- 仍保留 hiPhive 的工程优势：对称性处理、参数空间组织和下游输出兼容性。

### 与纯拟合法 workflow 对比

| 维度 | HIFINIT（MLFCS） | 纯拟合法（常见 hiPhive 用法） |
|---|---|---|
| 力常数来源 | 先有限差分直算，再投影到参数空间 | 由训练数据全局拟合参数 |
| ASR/对称约束 | 在 hiPhive 参数空间中显式施加，数值精度内严格满足 ASR 与对称约束（含平移/旋转相关约束） | 依赖拟合设置与数据质量，可能出现约束漂移或阶间误差再分配 |
| 阶间可分辨性 | 更强，低阶/高阶来源路径清晰 | 相对更弱，易出现阶间“吸收误差” |
| 计算成本 | 很高（高阶和大超胞下尤甚） | 相对更低（取决于数据规模与模型） |
| 计算器建议 | 强烈建议使用 ASE 机器学习势（NEP/MACE/DP/GAP 等） | DFT 与 MLP 均可，按场景选择 |

工程实践建议：
- 对三阶/四阶或更大体系，`HifinitRun` 通常应配合机器学习势函数；直接 DFT 全流程成本往往过高。
- 若目标是高精度且强约束（ASR/对称性）的一致满足，优先考虑 HIFINIT 路线。

## 重要说明（范围边界）

- MLFCS 的 `interface` 能力是对 I/O 与力解析的工程扩展。
- 它不会替代三阶/四阶核心重构算法本身。
- 可用接口名称取决于你本地安装的 phonopy 版本，建议先执行：

```bash
thirdorder interfaces
fourthorder interfaces
```

## 安装

```bash
git clone https://github.com/gtiders/mlfcs.git
cd mlfcs
pip install .
```

环境要求：
- Python `>= 3.12`
- 依赖见 [`pyproject.toml`](pyproject.toml)

## 快速开始

### 1) 三阶 CLI

```bash
# 生成位移超胞
thirdorder sow 4 4 4 --cutoff -3 --interface vasp --format vasp

# 对每个位移结构做外部力计算

# 回收并重构三阶力常数
thirdorder reap 4 4 4 --cutoff -3 \
  --interface vasp \
  --forces-interface vasp \
  --forces "./3RD_runs/*/vasprun.xml"
```

### 2) 四阶 CLI

```bash
fourthorder sow 3 3 3 --cutoff -2 --interface vasp --format vasp

fourthorder reap 3 3 3 --cutoff -2 \
  --interface vasp \
  --forces-interface vasp \
  --forces "./4TH_runs/*/vasprun.xml"
```

### 3) Python API：`ThirdOrderRun`（三阶，ASE 计算器）

```python
from mlfcs.thirdorder import ThirdOrderRun
from calorine.calculators import CPUNEP

calc = CPUNEP("nep.txt")
runner = ThirdOrderRun(na=3, nb=3, nc=3, cutoff=-3, structure_file="POSCAR")
runner.run_calculator(calc)  # 输出 FORCE_CONSTANTS_3RD
```

### 4) Python API：`FourthOrderRun`（四阶，ASE 计算器）

```python
from mlfcs.fourthorder import FourthOrderRun
from calorine.calculators import CPUNEP

calc = CPUNEP("nep.txt")
runner = FourthOrderRun(na=3, nb=3, nc=3, cutoff=-2, structure_file="POSCAR")
runner.run_calculator(calc)  # 输出 FORCE_CONSTANTS_4TH
```

### 5) Python API：`MLPHONON`（二阶）

```python
from ase.io import read
from calorine.calculators import CPUNEP
from mlfcs.secondorder import MLPHONON

prim = read("POSCAR")
calc = CPUNEP("nep.txt")

phonon = MLPHONON(
    structure=prim,
    calculator=calc,
    supercell_matrix=[2, 2, 2],
    kwargs_generate_displacements={"distance": 0.01},
)
phonon.run()
phonon.write("FORCE_CONSTANTS")  # text
phonon.write("fc2.hdf5")         # hdf5
```

### 6) Python API：`MLPSSCHA`

```python
from ase.io import read
from calorine.calculators import CPUNEP
from mlfcs.secondorder import MLPSSCHA

prim = read("POSCAR")
calc = CPUNEP("nep.txt")

sscha = MLPSSCHA(
    unitcell=prim,
    calculator=calc,
    supercell_matrix=[3, 3, 3],
    temperature=300,
    number_of_snapshots=1000,
    max_iterations=20,
    avg_n_last_steps=5,
    fc_output="fc2_sscha.hdf5",
    fc_output_format="hdf5",
)
sscha.run()
```

### 7) Python API：`HifinitRun`

```python
from ase.io import read
from calorine.calculators import CPUNEP
from mlfcs.hifinit import HifinitRun

prim = read("POSCAR")
supercell = read("SPOSCAR")
calc = CPUNEP("nep.txt")

runner = HifinitRun(
    primitive=prim,
    supercell=supercell,
    calculator=calc,
    displacement=0.005,
    cutoffs=[None, None, 4.0],
)
runner.run(out_dir="./hifinit_results", verbose=True)
```

## CLI 参数说明（`thirdorder` / `fourthorder`）

两者命令形态一致：

```bash
<tool> {sow|reap|interfaces} [na nb nc] [options]
```

常用参数：

| 参数 | 作用阶段 | 说明 |
|---|---|---|
| `na nb nc` | `sow`、`reap` | 超胞在 `a/b/c` 方向倍率 |
| `--cutoff` | `sow`、`reap` | 正数表示距离截断；负整数表示近邻壳层（例如 `-3`） |
| `-i`, `--input` | `sow`、`reap` | 输入结构文件（默认 `POSCAR`） |
| `--interface` | `sow`、`reap` | 结构读写接口名 |
| `--forces-interface` | `reap` | 力文件解析接口（默认跟随 `--interface`） |
| `--hstep` | `sow`、`reap` | 位移步长（单位 nm） |
| `--symprec` | `sow`、`reap` | 对称性精度 |
| `-f`, `--format` | `sow` | `vasp` 或 `same`（`same` 表示按 `--interface` 写出） |
| `--forces` | `reap` | 力文件列表或 glob 模式 |

注意：
- `reap` 要求力文件数量与期望位移数严格匹配。
- CLI `reap` 明确不支持 `.xyz` / `.extxyz` 轨迹输入。

## 输出文件

### `thirdorder` / `fourthorder`
- `FORCE_CONSTANTS_3RD`
- `FORCE_CONSTANTS_4TH`

### `MLPHONON` / `MLPSSCHA`
- 文本格式：`FORCE_CONSTANTS`
- HDF5 格式：`*.hdf5`

### `HifinitRun`（写入 `out_dir`）
- `potential.fcp`
- `FORCE_CONSTANTS_2ND`、`fc2.hdf5`
- `FORCE_CONSTANTS_3RD`、`fc3.hdf5`（当阶数 >= 3）
- `FORCE_CONSTANTS_4TH`（当阶数 >= 4）

## 最佳实践

- 在脚本中显式指定 `--interface` 与 `--forces-interface`，避免跨环境歧义。
- API 流程先用小超胞验证计算器稳定性，再放大体系。
- 手动循环调用 ASE 计算器并写轨迹时，建议先用 `SinglePointCalculator` 冻结结果，避免缓存导致的重复计算或数据混淆。

## 常见问题

### 能完全替代旧版 `thirdorder.py` / `fourthorder.py` 吗？
核心工作流概念兼容（`sow`/`reap`），但实现是工程化重构版本，并增加了显式接口控制与 Python API。

### 支持哪些 DFT 软件接口？
由 phonopy 在你本地环境可用的接口决定。请运行 `thirdorder interfaces` 查看实时列表。

### 可以直接接机器学习势吗？
可以。Python API 基于 ASE Calculator 抽象（示例使用 `calorine` 的 `CPUNEP`，也可替换为其他 ASE 兼容计算器）。

## 致谢

MLFCS 的思路与实现受以下项目启发并受益于其生态：
- `thirdorder.py`
- `fourthorder`
- `phonopy`
- `hiPhive`

感谢相关作者与维护者的长期工作。

## 许可证

本项目使用 **GNU General Public License v3.0**。详见 [`LICENSE`](LICENSE)。
