from hiphive import ForceConstantPotential
from hiphive.input_output.gpumd import write_r0
import os

# Read force constant potential
fcp = ForceConstantPotential.read('fcp_ols_original_model_5.fcp')
print(fcp)

# Construct 6x6x6 supercell
atoms = fcp.primitive_structure.copy()
supercell = atoms.repeat(6)

# Extract force constants for supercell
fcs = fcp.get_force_constants(supercell)

# Write potentials
tol = 1e-8
folder = 'ols_original_model_5_6x6x6_tol1e-08'
os.makedirs(folder, exist_ok=True)
for order in fcs.orders:
    fname1 = os.path.join(folder, 'fcs_order{}.in'.format(order))
    fname2 = os.path.join(folder, 'clusters_order{}.in'.format(order))
    fcs.write_to_GPUMD(fname1, fname2, order=order, tol=tol)

# Write reference positions
fname = os.path.join(folder, 'r0.in')
write_r0(fname, supercell)
