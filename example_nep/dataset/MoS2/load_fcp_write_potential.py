from hiphive import ForceConstantPotential
from hiphive.input_output.gpumd import write_r0
import os

# Read force constant potential
fcp = ForceConstantPotential.read('fcp_order6.fcp')
print(fcp)

# Construct 16x16x1 supercell
atoms = fcp.primitive_structure.copy()
supercell = atoms.repeat((16, 16, 1))

# Extract force constants for supercell
fcs = fcp.get_force_constants(supercell)

# Write potentials
tol = 1e-8
folder = 'order6_16x16x1_tol1e-08'
os.makedirs(folder, exist_ok=True)
for order in fcs.orders:
    fname1 = os.path.join(folder, 'fcs_order{}.in'.format(order))
    fname2 = os.path.join(folder, 'clusters_order{}.in'.format(order))
    fcs.write_to_GPUMD(fname1, fname2, order=order, tol=tol)

# Write reference positions
fname = os.path.join(folder, 'r0.in')
write_r0(fname, supercell)
