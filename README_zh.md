# fcs-order （中文指南）

提示：如果本项目对你有帮助，请为仓库点一个 Star。请先完整阅读本 README；如发现问题或有建议，请在 Issues 中提交反馈（建议附上复现步骤与环境信息）。

建议：命令行帮助会随版本更新，使用 `fcs-order --help` 与 `fcs-order <子命令> --help` 可获得更准确、最新的参数说明与示例。

说明：本项目的 `sow` 与 `reap` 命令与传统的 `thirdorder`/`fourthorder` 工具兼容，只是参数与 I/O 命名略有调整以实现更统一的接口。

基于 ASE 与 Typer 的力常数计算与自洽声子（SCPH）工具箱。支持计算二、三、四阶相互作用力常数，并可使用多种机器学习势（NEP、DeepMD、PolyMLP、MTP、TACE）或 Hiphive 作为计算后端。

仓库：<https://github.com/gtiders/fcs-order>

---

## 特性概览

- 二阶力常数（与 Phonopy FORCE_CONSTANTS 兼容）
- 三、四阶力常数（ShengBTE/文本格式）
- SCPH（自洽声子）工作流
- 多后端：NEP、DeepMD、PolyMLP、MTP、TACE，以及 Hiphive
- 面向 VASP 流程的结构生成（sow）与力常数重建（reap）
- 基于声子与 Monte Carlo 的结构扰动生成工具：`phonon_rattle` 与 `monte_rattle`
- 理论上也可配合任何 ASE 支持的第一性原理软件进行 sow/reap，只需按需调整参数与 I/O 设置（文件格式、路径顺序等）

---

## 安装

- Python 版本：>= 3.9, < 3.14
- 基础依赖在安装包中自动安装。按需安装后端依赖：
  - NEP: `pip install calorine`
  - DeepMD: `pip install deepmd-kit`
  - PolyMLP: `pip install pypolymlp`
  - Hiphive: `pip install hiphive`
  - TACE: `pip install tace`

提示：

- 若要计算二阶力常数（Phonopy 兼容的 `FORCE_CONSTANTS_2ND`），需要安装 Phonopy：`pip install phonopy`。
- 若要运行 SCPH 工作流，需要安装 Phonopy 与 Hiphive：`pip install phonopy hiphive`。

优先推荐通过 PyPI 安装：

```bash
pip install fcs-order
```

安装本项目（从 GitHub）：

```bash
pip install git+https://github.com/gtiders/fcs-order.git
```

命令入口：`fcs-order`（或通过 Python 运行模块入口）

```bash
fcs-order --help
```

---

## I/O 与基本约定

- 结构文件：任意 ASE 可读取格式（POSCAR/CONTCAR, CIF, XYZ/extxyz 等）；默认 `POSCAR`。
- 超胞规格：
  - mlp2/scph：使用 3 个整数（对角）或 9 个整数（3×3 矩阵）指定超胞矩阵；
  - sow/reap、mlp3、mlp4：使用三个位置参数 `na nb nc` 指定超胞重复数。
- 截断 `--cutoff/-c`：负数表示“按最近邻层数”；正数表示“按距离（nm）”。

---

## 快速上手（典型工作流）

1) 使用 sow 生成三阶/四阶位移结构与未位移结构：

```bash
# 以三阶为例（order=3），超胞 2×2×2，负号表示按近邻层数，输出为 VASP POSCAR
fcs-order sow 2 2 2 -c -6 -r 3 -p POSCAR -f poscar -o disps
# 生成：disps/3RD.SPOSCAR 与 disps/3RD.POSCAR.0001 ...
```

2) 用所选后端计算位移结构的受力，生成可被 ASE 读取的力文件（如 vasprun.xml/OUTCAR、extxyz 等）。确保文件顺序与 sow 生成的位移结构顺序一致。

3) 使用 reap 从受力文件重建三阶/四阶力常数：

```bash
# 三阶示例（order=3），列出所有受力文件，顺序必须一致
fcs-order reap 2 2 2 -c -6 -r 3 -p POSCAR disps/vasprun_*.xml
# 输出：FORCE_CONSTANTS_3RD
```

4) 二阶力常数可通过 mlp2 直接计算并输出 Phonopy 兼容的 `FORCE_CONSTANTS_2ND`。

5) 若需要进行 SCPH，自备二阶力常数或由后端在循环中评估，然后运行 `scph <backend>`。

---

## sow：生成位移结构

命令：`fcs-order sow na nb nc --cutoff <CUTOFF> [--order 3|4] [--poscar POSCAR] [--out-format poscar|vasp|cif|xyz] [--out-dir DIR] [--name-template ...] [--undisplaced-name ...]`

- 主要选项：
  - `-r/--order`：3（三阶）或 4（四阶），默认 3。
  - `-c/--cutoff`：负数=近邻层；正数=半径（nm）。
  - `-p/--poscar`：原胞结构文件，默认 POSCAR。
  - `-f/--out-format`：输出格式（默认 poscar）。
  - `-o/--out-dir`：输出目录。
  - 模板：`--name-template` 与 `--undisplaced-name` 支持占位符 `{order}`, `{phase}`, `{index}`, `{index_padded}`, `{width}`, `{ext}`。

输出示例（VASP 格式）：

- 未位移：`3RD.SPOSCAR` 或 `4TH.SPOSCAR`
- 位移：`3RD.POSCAR.0001`、`4TH.POSCAR.0001` 等

---

## reap：从受力文件重建力常数

命令：`fcs-order reap na nb nc --cutoff <CUTOFF> [--order 3|4] [--poscar POSCAR] VASP_RUN_FILES...`

- 受力文件需为 ASE 可读取格式（vasprun.xml/OUTCAR、extxyz 等）。
- 文件数量必须等于所需运行次数：
  - 三阶：`4 * N_irred`
  - 四阶：`8 * N_irred`
- 输出：`FORCE_CONSTANTS_3RD` 或 `FORCE_CONSTANTS_4TH`

---

## mlp2：二阶力常数（Phonopy 兼容）

命令组：`fcs-order mlp2 <backend>`，其中 `<backend>` ∈ {`nep`, `dp`, `ploymp`, `mtp2`, `tace`, `hiphive`}

- 通用参数（不同子命令的名称一致或相近）：
  - `supercell_matrix`：3 或 9 个整数（位置参数）。
  - `--poscar`：原胞结构文件，默认 POSCAR。
  - `--outfile/-o`：输出文件名，默认 `FORCE_CONSTANTS_2ND`。

- NEP（二阶）示例（可选 GPU）：

```bash
fcs-order mlp2 nep 2 2 2 -p nep.txt -g -o FORCE_CONSTANTS_2ND
```

- DP（二阶）：

```bash
fcs-order mlp2 dp 2 2 2 -p model.pb -o FORCE_CONSTANTS_2ND
```

- PolyMLP（二阶）：

```bash
fcs-order mlp2 ploymp 2 2 2 -p polymlp.pot -o FORCE_CONSTANTS_2ND
```

- MTP（二阶）（子命令名为 `mtp2`，需可用的 `mlp` 可执行程序）：

```bash
fcs-order mlp2 mtp2 2 2 2 -p pot.mtp --mtp-exe mlp -o FORCE_CONSTANTS_2ND
```

- TACE（二阶）（可选 `--device/--dtype/--level`）：

```bash
fcs-order mlp2 tace 2 2 2 -m model.ckpt -o FORCE_CONSTANTS_2ND --device cuda --dtype float64 --level 0
```

- Hiphive（二阶，从 fcp 输出到 Phonopy 文本）：

```bash
fcs-order mlp2 hiphive 2 2 2 -p potential.fcp
```

---

## mlp3：三阶力常数

命令组：`fcs-order mlp3 <backend>`，其中 `<backend>` ∈ {`nep`, `dp`, `ploymp`, `mtp2`, `tace`, `hiphive`}

- 通用参数：`na nb nc`、`--cutoff/-c`、`--poscar`（默认 POSCAR）、`--is-write`（保存中间位移结构/力）。

- NEP（三阶）：

```bash
fcs-order mlp3 nep 2 2 2 -c -6 -p nep.txt --is-write --is-gpu
```

- DP（三阶）：

```bash
fcs-order mlp3 dp 2 2 2 -c -6 -p model.pb
```

- PolyMLP（三阶）：

```bash
fcs-order mlp3 ploymp 2 2 2 -c -6 -p polymlp.pot
```

- MTP（三阶）（子命令名为 `mtp2`，需 `mlp`）：

```bash
fcs-order mlp3 mtp2 2 2 2 -c -6 -p pot.mtp --mtp-exe mlp --is-write
```

- TACE（三阶）：

```bash
fcs-order mlp3 tace 2 2 2 -c -6 -m model.ckpt --device cuda --dtype float32 --level 0
```

- Hiphive（三阶，从 fcp 直接导出到 ShengBTE 格式）：

```bash
fcs-order mlp3 hiphive 2 2 2 -p potential.fcp
```

输出：`FORCE_CONSTANTS_3RD`

---

## mlp4：四阶力常数

命令组：`fcs-order mlp4 <backend>`，其中 `<backend>` ∈ {`nep`, `dp`, `ploymp`, `mtp2`, `tace`, `hiphive`}

- 通用参数：`na nb nc`、`--cutoff/-c`、`--poscar`（默认 POSCAR）、`--is-write`。

提示：尽管进行了算法与实现优化，四阶计算仍可能占用较大的内存，建议在高内存环境中运行

- NEP（四阶）：

```bash
fcs-order mlp4 nep 2 2 2 -c -6 -p nep.txt --is-gpu
```

- DP（四阶）：

```bash
fcs-order mlp4 dp 2 2 2 -c -6 -p model.pb
```

- PolyMLP（四阶）：

```bash
fcs-order mlp4 ploymp 2 2 2 -c -6 -p polymlp.pot
```

- MTP（四阶）（子命令名为 `mtp2`）：

```bash
fcs-order mlp4 mtp2 2 2 2 -c -6 -p pot.mtp --mtp-exe mlp --is-write
```

- TACE（四阶，支持 `--device/--dtype/--level`）：

```bash
fcs-order mlp4 tace 2 2 2 -c -6 -m model.ckpt --device cuda --dtype float32 --level 0
```

- Hiphive（四阶，使用 fcp 与 supercell 求力常数）：

```bash
fcs-order mlp4 hiphive 2 2 2 -c -6 -p potential.fcp
```

输出：`FORCE_CONSTANTS_4TH`

---

## phonon_rattle：基于声子的热振动结构生成

命令：`fcs-order phonon_rattle SPOSCAR --fc2 FORCE_CONSTANTS -T 300 600 900 [选项]`

- 主要参数：
  - 位置参数 `SPOSCAR`：超胞结构（SPOSCAR/POSCAR 等 ASE 可读取格式）。
  - `--fc2`：二阶力常数文件路径，默认为 `FORCE_CONSTANTS`，格式与 Phonopy 文本 `FORCE_CONSTANTS` 兼容。
  - `-T/--T`：温度列表（K），例如 `-T 300 600 900`。
  - `-n/--n-structures`：每个温度下生成的结构数，默认 10。
  - `--qm-statistics/--no-qm-statistics`：是否使用量子简谐振子统计（默认关闭，为经典近似）。
  - `--imag-freq-factor`：处理虚频的因子，默认为 1.0；虚频会被替换为 `factor * |w^2|`。
  - `-f/--format`：输出格式，`vasp|cif|qe|xyz`，默认 `vasp`。
  - `-p/--prefix`：可选文件名前缀，若提供则所有温度与结构共享一个全局编号，例如 `prefix000.vasp`。
  - `--eps/--no-eps`：是否对每个结构施加体积涨落；配合 `--min-volume`、`--max-volume` 指定体积缩放区间。

- 文件命名：
  - 若未指定 `--prefix`，默认命名为 `phonon_rattle_T{T}_id{i}.vasp`（或相应扩展名）。
  - 指定前缀时，则为连续编号：`<prefix><index>.<ext>`。

- 此命令适合：在给定二阶力常数与温度下，生成满足谐振子统计的热振动结构，可配合 ML 势或第一性原理计算采样有限温度性质。

---

## monte_rattle：Monte Carlo 摇动结构生成

命令：`fcs-order monte_rattle SPOSCAR -n 50 --rattle-std 0.05 --d-min 1.0 [选项]`

- 主要参数：
  - 位置参数 `SPOSCAR`：超胞结构文件。
  - `-n/--n-structures`：生成结构数，默认 10。
  - `--rattle-std`：单步高斯位移的标准差（Å），默认 0.05。
  - `--d-min`：用于 Monte Carlo 接受率判据的最近邻距离阈值（Å）。
  - `--width`：误差函数宽度，控制 `d_min` 附近的接受概率平滑度，默认 0.1。
  - `--n-iter`：Monte Carlo 循环次数，迭代次数越多，总体位移越大，默认 10。
  - `--max-disp`：单个原子的最大允许位移模长，超过即拒绝该步，默认 2.0 Å。
  - `--seed`：随机种子，默认 42。
  - `-f/--format`：输出格式，`vasp|cif|qe|xyz`，默认 `vasp`。
  - `-p/--prefix`：可选文件名前缀，若提供则使用 `<prefix><index>.<ext>` 命名。
  - `--eps/--no-eps`：是否对每个结构施加随机体积涨落，体积比从 `[min_volume, max_volume]` 中均匀采样。

- 默认命名：若未指定前缀，文件名为 `mc_rattle_id{i}.vasp`（或对应扩展名）。
- Monte Carlo rattle 会拒绝使原子位移过大或最近邻距离过小的尝试，从而避免原子交换位点等异常结构。

---

## scph：自洽声子（SCPH）

命令组：`fcs-order scph <backend>`，其中 `<backend>` ∈ {`nep`, `dp`, `ploymp`, `mtp2`, `tace`, `hiphive`}

- 通用参数：
  - `supercell_matrix`：3 或 9 个整数（位置参数）。
  - `--poscar/-\\-poscar`：原胞结构文件，默认 POSCAR。
  - `--temperatures/-T`：如 "100,200,300"。
  - `--cutoff/-c`：簇空间截断半径。
  - `--alpha/-a`：SCPH 混合参数，默认 0.2。
  - `--n-iterations/-i`：迭代次数，默认 100。
  - `--n-structures/-n`：生成结构数，默认 50。
  - `--fcs-2nd/-F`：初始二阶力常数（可选）。
  - `--is-qm/-q`：是否使用量子统计（默认 True）。
  - `--imag-freq-factor/-I`：处理虚频的因子，默认 1.0。

> 说明：本项目的 SCPH 功能集成并调用了 Hiphive 的相关能力，更多信息参考 Hiphive 文档：<https://hiphive.materialsmodeling.org/>

- NEP（可选 GPU）：

```bash
fcs-order scph nep 2 2 2 -T 100,200,300 -c 4.5 -p nep.txt --poscar POSCAR --is-gpu
```

- DP：

```bash
fcs-order scph dp 2 2 2 -T 100,200,300 -c 4.5 -p graph.pb --poscar POSCAR
```

- TACE（可选 `--device/--dtype/--level`）：

```bash
fcs-order scph tace 2 2 2 -T 300 -c 4.5 -m model.ckpt --device cuda --dtype float32 --level 0
```

- Hiphive（使用 fcp）

```bash
fcs-order scph hiphive 2 2 2 -T 100,200,300 -c 4.5 -p model.fcp --poscar POSCAR
```

- PolyMLP（ploymp）：

```bash
fcs-order scph ploymp 2 2 2 -T 100,200,300 -c 4.5 -p polymlp.pot --poscar POSCAR
```

- MTP（子命令名为 `mtp2`，需 `mlp`）：

```bash
fcs-order scph mtp2 2 2 2 -T 100,200,300 -c 4.5 -p pot.mtp --poscar POSCAR --mtp-exe mlp
```

运行时会写出 `scph_SPOSCAR`，并在循环中评估力/力常数，随后进行收敛分析与导出。

---

## 常见问题（FAQ）

- 截断的单位与含义？
  - 负数：按最近邻层数；正数：按空间距离（单位：nm）。
- 受力文件顺序错乱如何处理？
  - `reap` 要求文件顺序与 `sow` 生成的位移结构顺序对应；建议程序内生成受力文件时保留编号。
- MTP 子命令为何命名为 `mtp2`？
  - 当前实现中 MTP 子命令名即为 `mtp2`，用于二阶/三阶/四阶与 SCPH 命令组；调用示例见上。
- Hiphive 使用注意事项？
  - 需要可读取的 fcp 文件，超胞尺寸需不小于 fcp 训练使用的超胞。

---

## 许可证

本项目采用 Apache-2.0 许可。详见 `LICENSE`。
