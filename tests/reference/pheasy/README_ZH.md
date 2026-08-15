# Pheasy 参考基准

[English](README.md)

这些基准使用 [Pheasy 仓库](https://gitlab.com/cplin/pheasy)公开的原始位移—力数据及力常数文件。
Pheasy 采用 GPL-3.0-only 许可证；MLFCS 不会把整个外部仓库复制进来。

这里明确区分两类比较：

- `tests/Si` 提供 10 个含 128 个原子的 Si 快照以及 FC2/FC3 参考文件，可以做同一仓库数据的
  直接交叉验证；它并不是论文中用于 FC2--FC6 的完整 64 个训练构型加 64 个测试构型。
- `examples/SrTiO3-QE` 提供 30 个含 40 个原子的构型，并明确给出了 FC2--FC6 设置。使用
  `reference_tools/benchmark_pheasy_fc6.py` 可由 MLFCS 联合拟合这五阶。

SrTiO3 设置为 FC2/FC3 不指定有限截断，FC4--FC6 截断 6 A，各阶最大作用体数为
`2, 3, 3, 2, 2`。Pheasy 对这个极性材料会先移除解析长程静电力，而 MLFCS 当前拟合输入的
总力，因此逐元素张量差异仅作为诊断；力重构误差、不变性残差和下游声子结果是更强的判据。

论文使用的独立数据集大于仓库公开的这些样例。论文中的力误差和输运结果属于文献基线，不能
直接作为这些较小公开数据集的通过阈值。

```bash
git clone --depth 1 https://gitlab.com/cplin/pheasy.git
uv run python reference_tools/benchmark_pheasy_fc6.py --pheasy-root pheasy
```
