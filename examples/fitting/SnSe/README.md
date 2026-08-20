# SnSe 拟合案例

本案例使用公开的 hiPhive force-constant potential 作为 ASE 计算器，生成 300 K 的
NVT 到 NVE 轨迹，然后用 MLFCS 拟合二阶、三阶和四阶力常数。拟合的最大 body order
分别为 FC2=2、FC3=3、FC4=4，截断半径分别为 8.0、6.5 和 6.5 Å。

## 目录

- `input/primitive.vasp`：primitive 结构；
- `input/reference.vasp`：由 MLFCS 的 phonopy 顺序扩包器生成的 `2x4x4` reference；
- `input/fcp_*.pickle`：公开 hiPhive 势函数；
- `reference/`：上游 GPUMD `4x11x11` 结构和热导率参考数据；
- `md/run.py`：300 K NVT/NVE 轨迹生成；
- `fit.py`：FC2/FC3/FC4 拟合；
- `plot.py`：使用 reference 超胞，让 phonopy 自动寻找 primitive 并绘制 seekpath 声子谱。

## 运行

```bash
uv run python examples/fitting/SnSe/input/make_supercell.py
uv run --with ase --with hiphive python examples/fitting/SnSe/md/run.py --overwrite
uv run python examples/fitting/SnSe/fit.py
uv run --with phonopy --with seekpath --with matplotlib python examples/fitting/SnSe/plot.py
```

拟合输出保存在 `results/fc234`。生成的力常数和拟合缓存只作为本地产物，不纳入案例源数据。
