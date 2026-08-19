# GPUMD input files and average results for MoS2

This folder contains the following files and scripts:
* `fcp_order6.fcp`, force constant potential that has been constructed using hiphive for a model that includes second, third, fourth, fifth and sixth order interactions up to 10.4, 5.6, 4.0, 2.5, and 2.5 Å, respectively.
* `load_fcp_write_potential.py`, script that demonstrates how the force constant potential can be loaded using hiphive and written as a potential with the format required by GPUMD
* `runs.txt/run.in`, example of a driver input file, which can be used to run a number of independent GPUMD simulations in sequence
* `run/run.in`, example of a GPUMD input file that specificies all parameters required to run a 4 ns long HNEMD simulation, using a 1 fs time step and a driving force of 4e-5, in the NVT ensemble with a NHC thermostat at 300 K.
* `run/xyz.in`, example of a GPUMD input file that describes the input structure in the form of a 16x16x1 MoS2 supercell.
* `run/order6_16x16x1_tol1e-08.txt`, example of an GPUMD driver input file for a force constant potential, which is found in the folder `order6_16x16x1_tol1e-08`, for a system with two atomic types, which also specifies that force constants up to the sixth order and third order heat currents should be employed
* `md-results-HNEMD.json`, file, in the form of a Pandas dataframe, with the average values, standard deviations and correlation lengths for the lattice thermal conductivities obtained from HNEMD simulations for MoS2 using various supercell sizes and driving forces at temperatures between 100 and 500 K.
* `md-results-EMD.json`, file, in the form of a Pandas dataframe, with the average values, standard deviations and correlation lengths for the lattice thermal conductivities obtained from EMD simulations for MoS2 using 16x16x1 supercells at temperatures between 100 and 600 K.

## MLFCS finite-difference conversion

`original/primitive.vasp` and `original/supercell.vasp` preserve the published FCP primitive cell and `16x16x1` GPUMD MD supercell. For phono3py FC2/FC3 work, the recommended finite-difference supercell is `4x4x1` (48 atoms), which is sufficient for the 10.4 Angstrom FC2 cutoff. Outputs are placed under `mlfcs/fc2` and `mlfcs/fc3` when generated.

The published EMD reference conductivity is:

| Temperature (K) | Conductivity (W/m K) |
| ---: | ---: |
| 300 | 134.412671 |
| 500 | 72.687814 +/- 3.453995 |
| 600 | 61.184535 +/- 4.597483 |
