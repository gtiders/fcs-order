# SSCHA 案例

本目录的结果由 K4As4Pt2 的 Polymlp 势函数在 300 K 重新生成。谐波力常数来自新有限差分案例的 `results/harmonic/mlfcs.h5`，SSCHA 每次使用 100 个快照并迭代 5 次；命令行可增加 `--iterations` 进行更长的收敛试验。

运行：

    python run_sscha.py
    python plot_bands.py

`results/sscha.h5`、`results/FORCE_CONSTANTS_SSCHA`、`results/history.json` 和 `figures/harmonic_vs_sscha.png` 都是本目录脚本新生成的文件。
