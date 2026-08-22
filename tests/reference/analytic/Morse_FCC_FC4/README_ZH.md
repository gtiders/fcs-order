# FCC Morse FC4 解析基准

[English](README.md) | 中文

该基准不依赖另一个力常数拟合器来验证四阶结果。ASE 使用 `MorsePotential` 弛豫单组分
FCC 晶胞；独立 JAX 实现对 Morse 对势做四次自动微分；MLFCS 则对 ASE calculator
产生的力做中心有限差分。

Ar 只是 ASE 元素标签，参数是约化单位数值基准而非氩势：`epsilon=1 eV`、`rho0=6`、
`r0=1 Angstrom`、切换区间 `1.15–1.30 Angstrom`，超胞为 3x3x3，MLFCS 截断为
`1.1 Angstrom`。解析路径固定周期最近邻键表，通过四层 `jax.jacfwd` 求导，不使用
MLFCS 的差分模板、对称性重建或 ASR。测试还要求位移步长减半后误差约缩小四倍，验证
中心差分的二阶收敛性。

```bash
uv run pytest tests/reference/analytic/Morse_FCC_FC4/test_morse_fc4.py
```
