# MLFCS 与 phonopy 的 KCl SSCHA 对比

[English](COMPARISON.md) | 中文

## 目的

本对比使用 phonopy 自身测试集维护的真实非谐机器学习势，检验完整的原生 MLFCS
路径。它比解析谐振采样测试更严格：势函数给出非线性能量和力，MLFCS 执行随机采样和
带约束 FC2 拟合，最后将张量与自由能同 phonopy 的验收值比较。

## 溯源

夹具来自 phonopy 官方仓库提交 `fb63319c071f264e01e1cd4d85a81526c6c7a40a`，
协议为 BSD-3-Clause：

- `test/polymlp_KCL-120.yaml`；
- `test/phonopy_KCl.yaml`；
- `example/KCl-SSCHA/phonopy_sscha_fc_JPCM2022.yaml.xz`。

pypolymlp 势由 120 个随机位移 KCl 结构训练。精确哈希和上游许可证保存在
[`data/`](data/) 中。

## 共同物理条件

| 设置 | MLFCS | phonopy 参考 |
|---|---:|---:|
| 材料 | KCl | KCl |
| 输入晶胞 | 8 原子常规胞 | 8 原子常规胞 |
| 超胞 | 2x2x2，64 原子 | 2x2x2，64 原子 |
| 温度 | 300 K | 300 K |
| 根随机种子 | 42 | 42 |
| 势函数 | 同一 `polymlp_KCL-120.yaml` | `polymlp_KCL-120.yaml` |
| 位移上限 | None | None |
| 统计 | 量子 | 量子 |

## 有意保留的数值工作量差异

phonopy 上游测试每轮使用 50 个快照，执行 3 轮 canonical 迭代。当前 WSL 环境重复运行
该完整工作量时，进程会因 JAX 和 pypolymlp 的累计内存压力被系统终止。因此串行 CI
参考使用 10 个快照和 1 轮 canonical 迭代。它足以检验完整 API 和物理量级，但不宣称
这是已收敛的生产计算。

谐振自由能的约定也不完全相同。phonopy 在稠密倒空间网格上计算谐振项，当前原生
MLFCS 使用与采样超胞相容的 q 点。因此 FC2 是更干净的主要检查；自由能容差同时覆盖
有限采样和 q 网格效应。

## 结果

| 物理量 | MLFCS CI 参考 | phonopy 测试参考 | 解释 |
|---|---:|---:|---|
| 初始化 K 自作用 FC2 | `1.9042 eV/Angstrom^2` | 上游未设验收值 | 初始拟合稳定 |
| canonical K 自作用 FC2 | `2.1625 eV/Angstrom^2` | `2.1 +/- 0.1 eV/Angstrom^2` | 位于官方范围内 |
| 每原胞自由能 | `-0.0949 eV` | `-0.0986 +/- 0.001 eV` | 相差约 `3.7 meV` |

MLFCS 输入对象是含 4 个原胞的常规胞，因此与 phonopy 的每原胞数值比较前，需将 MLFCS
报告的自由能除以 4。

## 本对比确认的语义

- 笛卡尔初始化构型不是从拟合谐振哈密顿量采样，因此不定义 SSCHA 自由能；MLFCS 在
  该轮报告 `None`；
- 每轮 canonical 迭代都由根种子和迭代号派生不同子种子；整次运行可复现，不同轮不会复用
  同一随机流；
- 默认不执行最大位移裁剪，因此不会静默截断采样分布。

## 能证明什么

本测试证明，phonopy 官方非线性 KCl 势可以完整通过 MLFCS 的随机采样、ASE calculator、
Gram FC2 拟合、ASR 和自由能路径，并恢复 phonopy 验收的 FC2 量级。

它不证明两种 SSCHA 实现已完全收敛且逐元素相等。这样的结论需要对齐快照数、迭代收敛、
倒空间网格、拟合约束和统计误差。仓库已保存已发表 FC2，便于未来做能带和模态分辨对比。
