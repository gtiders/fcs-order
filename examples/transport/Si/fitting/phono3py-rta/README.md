# Si 拟合力常数 phono3py RTA

input/ 中的 FC2 和 FC3 是从新生成的联合拟合结果复制得到的运行输入。默认使用 11x11x11 网格，
结果写入 results/。

    uv run python run_rta.py --mesh 11 11 11 --temperatures 300 400 500 600 700 800 900
    uv run python analyze.py
