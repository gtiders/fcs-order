# K4As4Pt2 有限差分

本案例使用 `input/polymlp.yaml`，原胞和参考超胞分别为 `primitive.vasp` 与 `supercell.vasp`，扩包矩阵为 `2 x 2 x 3`，位移为 0.01 Angstrom。二阶没有截断，三阶截断为 12 Bohr。

运行 `uv run --with pypolymlp --with phono3py python run.py --route both` 会分别执行 MLFCS 和 phono3py 有限差分路线。力评估和力常数写入 `results/`，其中 MLFCS 结果保留原生 HDF5、phonopy 二阶和 ShengBTE 三阶，phono3py 路线保留配套 HDF5。

运行 `python compare_legacy.py` 可逐数据集核对新生成的力评估和 HDF5 结果。
