# K4As4Pt2 loop-SCPH

先在拟合案例生成三体四阶结果，再运行：

    uv run python run.py --temperatures 300 600 900 --max-iterations 100 --overwrite

脚本使用 reference 超胞派生的 q 点：默认 `interpolation_multiplier=1`、
`scph_multiplier=2`，结果写入 `results/`。`plot_bands.py` 单独读取这些新结果并绘制声子谱。
