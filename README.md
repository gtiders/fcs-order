# fcs-order

Force constants calculation toolkit built on ASE and Typer. Compute 2nd/3rd/4th-order interatomic force constants and run SCPH workflows using various machine-learning potentials (NEP, DeepMD, PolyMLP, MTP) or Hiphive.

Repo: https://github.com/gtiders/fcs-order

## Features

- 2nd-order force constants (phonopy-compatible)
- 3rd- and 4th-order force constants (ShengBTE/ShengBTE-like formats)
- SCPH (Self-Consistent Phonon) workflow
- Multiple ML calculators: NEP, DeepMD, PolyMLP, MTP; and Hiphive
- Handy sow/reap utilities for VASP workflows

## Installation

- Python 3.9+
- Recommended: create a fresh virtual environment

```bash
pip install git+https://github.com/gtiders/fcs-order.git
```

Optional dependencies per backend (install only what you need):

- NEP: `pip install calorine`
- DeepMD: `pip install deepmd-kit`
- PolyMLP: `pip install pypolymlp`
- Hiphive: `pip install hiphive`
- MTP: external `mlp` binary (Moment Tensor Potential), make sure it’s on PATH

## CLI Overview

The main entry is a Typer App with subcommands:

- Top-level utilities
  - `sow3` / `sow4`: generate displaced POSCARs
  - `reap3` / `reap4`: collect forces (from VASP) and build IFCs
  - `plot_phband`: plot phonon band structures from FORCE_CONSTANTS files
  - `phonon_sow`: rattle utility (from utils)
- Sub-apps for ML potentials and SCPH
  - `mlp2`: 2nd-order IFCs via ML calculators (nep/dp/ploymp/mtp)
  - `mlp3`: 3rd-order IFCs via ML calculators (nep/dp/ploymp/mtp)
  - `mlp4`: 4th-order IFCs via ML calculators (nep/dp/ploymp/mtp)
  - `scph`: SCPH workflow via ML calculators (nep/dp/ploymp/hiphive/mtp)

Run `python -m fcsorder --help` for the full tree.

## Commands

### sow3
Generate 3rd-order displacement structures.

```bash
python -m fcsorder sow3 NA NB NC --cutoff <CUTOFF> --poscar POSCAR
```

### sow4
Generate 4th-order displacement structures.

```bash
python -m fcsorder sow4 NA NB NC --cutoff <CUTOFF> --poscar POSCAR
```

### reap3
Collect VASP results and build 3rd-order IFCs.

```bash
python -m fcsorder reap3 NA NB NC --cutoff <CUTOFF> [--is-sparse] --poscar POSCAR VASPRUN1.xml VASPRUN2.xml ...
```

### reap4
Collect VASP results and build 4th-order IFCs.

```bash
python -m fcsorder reap4 NA NB NC --cutoff <CUTOFF> [--is-sparse] --poscar POSCAR VASPRUN*.xml
```

### mlp2 (2nd-order)
Subcommands: `nep`, `dp`, `ploymp`, `mtp`

Matrix input for supercell: either 3 diagonal ints or 9 ints for a full 3x3 matrix.

- NEP
```bash
python -m fcsorder mlp2 nep 2 2 2 --potential nep.txt --poscar POSCAR --outfile FORCE_CONSTANTS_2ND [--is-gpu]
```
- DeepMD
```bash
python -m fcsorder mlp2 dp 2 2 2 --potential model.pb --poscar POSCAR --outfile FORCE_CONSTANTS_2ND
```
- PolyMLP
```bash
python -m fcsorder mlp2 ploymp 2 2 2 --potential polymlp.pot --poscar POSCAR --outfile FORCE_CONSTANTS_2ND
```
- MTP (Moment Tensor Potential, requires `mlp` executable)
```bash
python -m fcsorder mlp2 mtp 2 2 2 --potential pot.mtp --poscar POSCAR [--mtp-exe mlp] --outfile FORCE_CONSTANTS_2ND
```
Note: For MTP, the code automatically detects unique elements from the POSCAR via ASE; you do not need to pass them.

### mlp3 (3rd-order)
Subcommands: `nep`, `dp`, `ploymp`, `mtp`

```bash
# Example with MTP
python -m fcsorder mlp3 mtp 2 2 2 --cutoff 3.0 --potential pot.mtp --poscar POSCAR [--mtp-exe mlp] [--is-write] [--is-sparse]
```

Other backends follow the same pattern as `mlp2`, with `--cutoff` required.

### mlp4 (4th-order)
Subcommands: `nep`, `dp`, `ploymp`, `mtp`

```bash
python -m fcsorder mlp4 mtp 2 2 2 --cutoff 3.0 --potential pot.mtp --poscar POSCAR [--mtp-exe mlp] [--is-write] [--is-sparse]
```

### scph
Subcommands: `nep`, `dp`, `hiphive`, `ploymp`, `mtp`

Common arguments:

- `primcell`: path to primitive cell (e.g. POSCAR)
- `supercell_matrix`: 3 or 9 ints
- `temperatures`: e.g. "100,200,300"
- `cutoff`: cluster space cutoff (backend-specific meaning)

Examples:

- NEP
```bash
python -m fcsorder scph nep POSCAR 2 2 2 --temperatures 100,200,300 --cutoff 3.0 --potential nep.txt [--is-gpu]
```
- DP
```bash
python -m fcsorder scph dp POSCAR 2 2 2 --temperatures 100,200,300 --cutoff 3.0 --potential graph.pb
```
- Hiphive
```bash
python -m fcsorder scph hiphive POSCAR 2 2 2 --temperatures 300 --cutoff 3.0 --potential model.fcp
```
- PolyMLP
```bash
python -m fcsorder scph ploymp POSCAR 2 2 2 --temperatures 100,200,300 --cutoff 3.0 --potential polymlp.pot
```
- MTP
```bash
python -m fcsorder scph mtp POSCAR 2 2 2 --temperatures 100,200,300 --cutoff 3.0 --potential pot.mtp [--mtp-exe mlp]
```
Note: For MTP, unique elements are obtained from the provided `primcell` via ASE.

## MTP Notes

- Requires `mlp` binary accessible via `PATH` or specify `--mtp-exe`.
- Temporary files are written under the system temp directory by default.
- In calculations, unique chemical symbols are collected from the ASE Atoms object automatically.

## Development

- Run formatting and basic lint if desired
- Contribute via PRs on GitHub: https://github.com/gtiders/fcs-order

## License

TBD.
