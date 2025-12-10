# fcs-order (中文文档)
需要修复计算器使用超饱是哪一个保证正确
fcs-order 是一个用于计算晶格热导率和声子性质的工具包，支持二阶、三阶和四阶力常数 (FCs) 的计算。它集成了多种机器学习势 (MLP) 和自洽声子 (SCPH) 计算功能。

> **注意**: CLI 已统一为单个 `fcsorder` 命令。旧的工作流命令（如 `sow`/`reap`）已被统一的 `fcX` 命令取代，新命令可一步完成位移生成、力计算（通过 MLP）和力常数重构。

## 安装

```bash
pip install fcs-order
```

**后端支持 (按需安装):**
- **NEP**: `pip install calorine`
- **DeepMD**: `pip install deepmd-kit`
- **PolyMLP**: `pip install pypolymlp`
- **Hiphive**: `pip install hiphive`
- **TACE**: `pip install tace`
- **Phonopy**: `pip install phonopy` (计算 fc2/scph 必需)
- **MACE**: 请参考mace的官方文档

## CLI 使用说明

主程序入口为 `fcsorder` (或 `forcekit`)。

```bash
fcsorder --help
```

### 1. 二阶力常数 (`fc2`)

使用 Phonopy 和机器学习势计算谐性力常数。

```bash
fcsorder fc2 [NA] [NB] [NC] -c <calculator> -p <potential> [选项]
```

**参数:**
- `NA`, `NB`, `NC`: 超胞扩胞倍数 (例如 `2 2 2`)。
- `-c, --calculator`: 后端类型 (`nep`, `dp`, `tace`, `hiphive` 等)。
- `-p, --potential`: 势函数模型文件路径。
- `-s, --structure`: 输入结构文件 (默认: `POSCAR`)。
- `-o, --output`: 输出文件名 (默认: `FORCE_CONSTANTS`)。
- `-f, --output-format`: `text` (Phonopy 格式) 或 `hdf5`。

**示例:**
```bash
fcsorder fc2 2 2 2 -c nep -p nep.txt
```

### 2. 三阶力常数 (`fc3`)

计算三阶非谐力常数。

```bash
fcsorder fc3 [NA] [NB] [NC] -c <calculator> -p <potential> -k <cutoff> [选项]
```

**参数:**
- `-k, --cutoff`: 截断半径。负数 (如 `-3`) 表示最近邻层数，正数 (如 `0.5`) 表示以 nm 为单位的距离。
- `-f, --output-format`: `text` (默认，输出 `FORCE_CONSTANTS_3RD`) 或 `hdf5` (通过 Hiphive 输出 `fc3.hdf5`)。
- `-w, --save-intermediate`: 是否保存中间的位移结构和受力文件。
- `--device`: 计算设备 `cpu` 或 `cuda`。

**示例:**
```bash
# 计算并导出 Phono3py HDF5 格式
fcsorder fc3 2 2 2 -c nep -p nep.txt -k -3 -f hdf5
```

### 3. 四阶力常数 (`fc4`)

计算四阶非谐力常数。

```bash
fcsorder fc4 [NA] [NB] [NC] -c <calculator> -p <potential> -k <cutoff> [选项]
```

**参数:**
- 与 `fc3` 类似。
- **注意**: 仅支持文本输出 (`FORCE_CONSTANTS_4TH`)，已移除 HDF5 支持。

**示例:**
```bash
fcsorder fc4 2 2 2 -c dp -p full.pb -k -2
```

### 4. 自洽声子 (`scph`)

使用 SCPH 方法进行非微扰声子重整化计算。

```bash
fcsorder scph [NA] [NB] [NC] -c <calculator> -p <potential> -T <temperatures> -k <cutoff> [选项]
```

**参数:**
- `-T, --temperatures`: 逗号分隔的温度列表 (例如 `100,300`)。
- `-k, --cutoff`: 簇展开截断半径 (nm)。
- `-a, --alpha`: 混合参数 (默认: 0.2)。
- `-i, --num-iterations`: 最大迭代次数 (默认: 30)。
- `-n, --num-structures`: 每次迭代生成的结构数 (默认: 500)。
- `-f, --output-format`: 有效力常数的输出格式 (`text` 或 `hdf5`)。

**示例:**
```bash
fcsorder scph 2 2 2 -c nep -p nep.txt -T 300 -k 4.5
```

## 结构生成工具

### 声子热振动 (`phonon-rattle`)
基于谐性声子生成热无序结构。
```bash
fcsorder phonon-rattle POSCAR --force-constants-file FORCE_CONSTANTS -T 300 -n 10
```

### 蒙特卡洛振动 (`monte-rattle`)
使用 MC rattle 生成结构，避免原子距离过近。
```bash
fcsorder monte-rattle POSCAR -n 10 --d-min 1.0 --rattle-amplitude 0.05
```

### 简单振动 (`rattle`)
使用不相关的的高斯噪声生成随机位移结构。
```bash
fcsorder rattle POSCAR -n 10 --rattle-amplitude 0.05
```

## 许可证
Apache-2.0
