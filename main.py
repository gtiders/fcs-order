from ase.io import read, write
from calorine.calculators import CPUNEP
from mlfcs.thirdorder import calculate_force_constants

poscar = read("/home/gwins/code_space/mlfcs/test/K3Au3Sb2/POSCAR")
calc = CPUNEP("/home/gwins/code_space/mlfcs/test/K3Au3Sb2/nep.txt")
myxyz = read("/home/gwins/code_space/mlfcs/test/K3Au3Sb2/my.xyz", index=":")
# calculate_force_constants(poscar, 3, 3, 3, -8, calculator=calc)

for i in myxyz:
    i.calc = calc
    i.get_forces()

write("my_forces.xyz", myxyz, format="extxyz")
