# K4As4Pt2 fitting

`source/prepare_dataset.py` converts the original ALAMODE `DFTSET_RAND` into
the strict ASE `train.extxyz` format used by MLFCS. The source data use Bohr and
Rydberg/Bohr and are converted to Angstrom and eV/Angstrom.

The legacy `../fiting/` directory is retained as provenance for the original
ALAMODE input and reference outputs. It is not used by these MLFCS runs.

The first anharmonic run limits FC4 to three distinct atomic sites:

```bash
uv run python source/prepare_dataset.py
uv run python anharmonic/run.py --body-order-4 3
uv run python thermal-conductivity/run_rta.py --fit three-body --temperature 300
```

The fitting script writes the native `mlfcs.h5`, phonopy `FORCE_CONSTANTS_2ND`,
ShengBTE `FORCE_CONSTANTS_3RD`, and `FORCE_CONSTANTS_4TH`. The RTA script reads
the matching FC2/FC3 HDF5 files and uses an 11 x 11 x 11 mesh. FC4 is not used
by phono3py's three-phonon thermal-conductivity calculation.

After the three-body run is verified, run `--body-order-4 4`; its results are
kept in the separate `anharmonic/four-body` directory.

The three-body FC4 result can also be used for the independent quartic
loop-SCPH case. The internal loop sum uses a 6 x 6 x 6 grid and the effective
dynamical matrices are solved on a 3 x 3 x 3 grid:

```bash
uv run python scph/run.py --temperatures 300 600 900 --max-iterations 100 --overwrite
uv run --with matplotlib python scph/plot.py
```

Each temperature directory contains the effective native FC2, phonopy text
FC2, q-grid frequencies, and the SCPH convergence history.

Convergence is measured by the root-mean-square frequency change over
interpolation-grid modes. It must be below `1e-10` THz and all squared
frequencies must be non-negative. The 600 K calculation does not converge in
100 iterations: its final RMS change is 2.22e-2 THz and its minimum frequency
is -1.782 THz. The output must therefore not be treated as a production
temperature-renormalized IFC. It is retained because it exercises the complete
FC4 contraction, dual meshes, effective-FC2 writer, and phonopy/SeeK-path
postprocessing.
