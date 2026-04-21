from mlfcs.thirdorder import ThirdOrderRun
from mlfcs.fourthorder import FourthOrderRun
from mlfcs.hifinit import HifinitRun
from mlfcs.secondorder import MLPHONON, MLPSSCHA
from ase.io import read
# 假设您使用 calorine 的 CPUNEP 计算器，也可以是任何 ASE Calculator
from calorine.calculators import CPUNEP

# 初始化运行器
# 参数: na=3, nb=3, nc=1, cutoff=-8 (第8近邻)
runner = ThirdOrderRun(3, 3, 1, -8)
fourthorder_runner = FourthOrderRun(3, 3, 1, -3)

# 定义 ASE 计算器
calc = CPUNEP("nep.txt")

# 直接运行计算，无需手动处理文件 I/O
runner.run_calculator(calc)
fourthorder_runner.run_calculator(calc)

# # hifinit 单类接口：准备结构并执行完整流程
prim = read("POSCAR")
supercell = read("SPOSCAR")
hifinit_runner = HifinitRun(
    primitive=prim,
    supercell=supercell,
    calculator=calc,
    displacement=0.005,
    cutoffs=[None, None, 4.0],
)
hifinit_runner.run(out_dir="./hifinit_results", verbose=True)

# secondorder: phonon
phonon_runner = MLPHONON(
    structure=prim,
    calculator=calc,
    supercell_matrix=[3, 3, 1],
    kwargs_generate_displacements={"distance": 0.01},
)
phonon_runner.run()
phonon_runner.write("fc2.hdf5")

# secondorder: sscha (small demo settings)
sscha_runner = MLPSSCHA(
    unitcell=prim,
    calculator=calc,
    supercell_matrix=[3, 3, 1],
    temperature=300,
    number_of_snapshots=100,
    max_iterations=10,
    avg_n_last_steps=5,
    fc_output="FORCE_CONSTANTS_SSCHA.hdf5",
    fc_output_format="hdf5",
    log_level=1,
)
sscha_runner.run()