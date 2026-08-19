# GPUMD input files and average results for SnSe

This folder contains the following files and scripts:
* `fcp_cm16_rfe-ridge_nf-3000_alpha-1.0.pickle`, force constant potential that has been constructed using hiphive for a model that includes two-body (three-body) second, third, fourth, fifth and sixth order interactions up to 8.0 (4.5), 6.5 (6.5), 6.5 (4.5), 4.0 (0), and 4.0 (0) Å, respectively.
* `load_fcp_write_potential.py`, script that demonstrates how the force constant potential can be loaded using hiphive and written as a potential with the format required by GPUMD
* `runs.txt/run.in`, example of a driver input file, which can be used to run a number of independent GPUMD simulations in sequence
* `run/run.in`, example of a GPUMD input file that specificies all parameters required to run a 1 ns long EMD simulation, using a 1 fs time step, in the NVT ensemble with a NHC thermostat at 300 K.
* `run/xyz.in`, example of a GPUMD input file that describes the input structure in the form of a 4x11x11 SnSe supercell.
* `run/fcp_cm16_rfe-ridge_nf-3000_alpha-1.0_4x11x11_tol1e-08.txt`, example of an GPUMD driver input file for a force constant potential, which is found in the folder `fcp_cm16_rfe-ridge_nf-3000_alpha-1.0_4x11x11_tol1e-08`, for a system with two atomic types, which also specifies that force constants up to the sixth order and third order heat currents should be employed
* `md-results.json`, file, in the form of a Pandas dataframe, with the average values, standard deviations and correlation lengths for the lattice thermal conductivities obtained from EMD simulations for SnSe using 4x11x11 supercells at temperatures between 100 and 400 K.

## MLFCS finite-difference conversion

`original/primitive.vasp` and `original/supercell.vasp` preserve the published FCP primitive cell and `4x11x11` GPUMD MD supercell. `mlfcs/fd_supercell.vasp` is the `2x4x4` (256 atom) finite-difference reference cell used for FC2/FC3. Outputs are placed under `mlfcs/fc2` and `mlfcs/fc3`.

The finite-difference force constants were generated directly from the published hiPhive FCP, without regenerating or modifying the training data:

```bash
uv run --with ase --with hiphive --with phonopy --with phono3py \
  python example_nep/run_fcp_finite_difference.py SnSe --overwrite
```

Plot the phonon bands and run phono3py RTA at 300 K with a `16x16x16` mesh. The
RTA script enables natural-isotope scattering and uses a 1 micrometer boundary
mean free path by default:

```bash
uv run --with seekpath --with matplotlib --with phonopy \
  python example_nep/dataset/SnSe/plot.py
.venv/bin/python example_nep/dataset/SnSe/run_rta.py
```

Override the boundary length with `--boundary-mfp MICROMETER`, or disable
isotope scattering with `--no-isotope`.

The RTA outputs are in `mlfcs/rta/kappa-rta.txt` and `mlfcs/rta/kappa-rta.npz`. The 300 K result is:

| Temperature (K) | kappa_xx | kappa_yy | kappa_zz (W/m K) |
| ---: | ---: | ---: | ---: |
| 300 | 0.937107 | 2.352428 | 2.001811 |

The published EMD reference conductivity is:

| Temperature (K) | Conductivity (W/m K) |
| ---: | ---: |
| 300 | 0.555661 +/- 0.065161 |
| 500 | not provided |
| 600 | not provided |
