# Si 有限差分 phono3py RTA

input/ 中的 FC2、FC3 和 reference supercell 来自有限差分案例。脚本使用 phono3py API，默认网格为
11x11x11，输出写入 results/。

    uv run python run_rta.py --mesh 11 11 11 --temperatures 300

reference/ 中的 ShengBTE 和 phonopy/thirdorder 数据只用于结果核对。
