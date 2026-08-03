"""External sow/reap outline; the user decides how each structure is evaluated."""

import numpy as np
from ase.io import read

from mlfcs import ForceConstantCalculation

primitive = read("POSCAR")
calculation = ForceConstantCalculation(
    primitive,
    order=3,
    supercell=(3, 3, 3),
    cutoff=-6,
    displacement=0.01,
)

structures = calculation.sow(atom_order="grouped")
# Submit `structures` in this exact order. Replace this placeholder with forces
# read from completed calculations; the required shape is printed below.
print(len(structures), calculation.plan.hash)
print((len(structures), len(calculation.supercell), 3))

forces = np.load("forces.npy")
force_constants = calculation.reap(
    forces,
    atom_order="grouped",
    plan_hash=calculation.plan.hash,
    acoustic_sum_rule=True,
)
force_constants.write("FORCE_CONSTANTS_3RD", format="shengbte", order=3)
