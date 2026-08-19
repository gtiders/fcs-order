# GPUMD input files and average results for BGG

This folder contains the following files and scripts:
* `fcp_ols_original_model_5.fcp`, force constant potential that has been constructed using hiphive for a model that includes second, third and fourth order interactions up to 5.4, 4.7, and 4.7 Å, respectively.
* `load_fcp_write_potential.py`, script that demonstrates how the force constant potential can be loaded using hiphive and written as a potential with the format required by GPUMD
* `runs.txt/run.in`, example of a driver input file, which can be used to run a number of independent GPUMD simulations in sequence
* `run/run.in`, example of a GPUMD input file that specificies all parameters required to run a 5 ns long HNEMD simulation, using a 1 fs time step and a driving force of 1e-4, in the NVT ensemble with a NHC thermostat at 100 K. The spectral heat current, PDOS, trajectories as well as the potential and kinetic contributions to the heat currents are included in the output.
* `run/xyz.in`, example of a GPUMD input file that describes the input structure in the form of a 6x6x6 BGG supercell. The atoms have been arranged into multiple groups which either includes all sites or corresponds to a specific element and Wyckoff site.
* `run/ols_original_model_5_6x6x6_tol1e-08.txt`, example of an GPUMD driver input file for a force constant potential, which is found in the folder `ols_original_model_5_6x6x6_tol1e-08`, for a system with three atomic types, which also specifies that force constants up to the fourth order and third order heat currents should be employed
* `md-results-HNEMD.json`, file, in the form of a Pandas dataframe, with the average values, standard deviations and correlation lengths for the lattice thermal conductivities obtained from HNEMD simulations for BGG using various supercell sizes and driving forces at temperatures between 50 and 300 K.
* `md-results-EMD.json`, file, in the form of a Pandas dataframe, with the average values, standard deviations and correlation lengths for the lattice thermal conductivities obtained from EMD simulations for BGG using 6x6x6 supercells at 100, 200, and 300 K.

## MLFCS finite-difference conversion

`original/primitive.vasp` and `original/supercell.vasp` are generated from the published hiPhive FCP. The MLFCS finite-difference reference supercell is `2x2x2` (432 atoms); its FC2/FC3 outputs, when generated, are placed under `mlfcs/fc2` and `mlfcs/fc3`.

The BaGaGe RTA loading test is in `run_rta.py`. It uses a `3x3x3` mesh at 300 K,
with natural-isotope scattering and a 1 micrometer boundary mean free path:

```bash
.venv/bin/python example_nep/dataset/BaGaGe/run_rta.py
```

Use `--boundary-mfp MICROMETER` or `--no-isotope` to change the scattering
settings.

The 3x3x3 loading/RTA test completed successfully. At 300 K it gives
`kappa_xx = 0.284984`, `kappa_yy = 0.284984`, and `kappa_zz = 0.284984`
W/(m K); the full tensor is in `mlfcs/rta/kappa-rta.txt`.

The published EMD reference conductivity is:

| Temperature (K) | Conductivity (W/m K) |
| ---: | ---: |
| 300 | 0.902699 +/- 0.085010 |
| 500 | not provided |
| 600 | not provided |
