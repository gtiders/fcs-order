"""Minimal direct-calculator FC2 example using ASE's built-in EMT model."""

from ase.build import bulk
from ase.calculators.emt import EMT

from mlfcs import ForceConstantCalculation

aluminum = bulk("Al", "fcc", a=4.05)
calculation = ForceConstantCalculation(
    aluminum,
    order=2,
    supercell=(2, 2, 2),
    cutoff=-2,
    displacement=0.01,
)

force_constants = calculation.run(EMT(), acoustic_sum_rule=True)
force_constants.write("FORCE_CONSTANTS_2ND", format="phonopy", order=2)

print(force_constants.orders)
print(force_constants.metadata)
