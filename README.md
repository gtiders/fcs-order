The fcs-order project is currently under active development. If you want to experience and use the project, you can install it via Git with the command `pip install git+https://github.com/gtiders/fcs-order.git`. After installation, run `fcsorder --help` to view detailed help information for each command, so that you can better understand and use the project's features.
# fcs-order

Force constants calculation toolkit built on ASE and Typer. Compute 2nd/3rd/4th-order interatomic force constants and run SCPH workflows using various machine-learning potentials (NEP, DeepMD, PolyMLP, MTP) or Hiphive.

Repo: https://github.com/gtiders/fcs-order

## User Guide

fcs-order is a CLI toolkit for computing interatomic force constants (2nd/3rd/4th order) and running SCPH workflows using ASE‑readable structures and various ML/physics backends.

### Installation

Requirements: Python 3.10–3.13 recommended.

```bash
pip install git+https://github.com/gtiders/fcs-order.git
```

Optional backends (install as needed):
- NEP: `pip install calorine`
- DeepMD: `pip install deepmd-kit`
- PolyMLP: `pip install pypolymlp`
- Hiphive: `pip install hiphive`
- MTP: external `mlp` binary in PATH

### I/O and formats

- Structures: any ASE‑readable file (POSCAR/CONTCAR, CIF, XYZ/extxyz, …)
- Forces: any ASE‑readable file with forces (vasprun.xml/OUTCAR, extxyz, …)
- All structure reads go through the internal I/O abstraction to keep behavior consistent.

### Supercell specification

- 3 integers for diagonal expansion (na nb nc), or 9 integers for a full 3×3 matrix (flattened).
- Used by mlp2/scph; mlp3/mlp4 use na nb nc positional arguments.

### Common short options

- `--poscar, -p` Structure path (default: POSCAR)
- `--cutoff, -c` Cutoff distance/value (semantics by command)
- `--potential, -P` Backend potential/model path
- `--temperatures, -T` e.g., "100,200,300"
- See more in the quick reference: `docs/cli_usage_en.md`.

### Quick start

```bash
# sow (3rd‑order) with CIF input and XYZ outputs
fcsorder sow 2 2 2 -c -6 -p prim.cif -f xyz -o out_dir

# reap (3rd‑order) with extxyz forces
fcsorder reap 2 2 2 -c -6 pos/disp_*.xyz -p prim.cif

# mlp3 with NEP backend
fcsorder mlp3 nep 2 2 2 -c -6 -P nep.txt -p POSCAR

# scph with DP backend
fcsorder scph dp 2 2 2 -p prim.cif -T "100,200,300" -c 4.5 -P graph.pb -A
```

### Commands overview

- sow: generate displaced structures for IFC calculations (order 3/4 via `-r`)
- reap: reconstruct IFCs from forces in ASE‑readable files
- mlp2: compute 2nd‑order IFCs (subcommands: nep/dp/ploymp/mtp), supercell matrix input
- mlp3: compute 3rd‑order IFCs (subcommands: nep/dp/ploymp/mtp), uses na nb nc
- mlp4: compute 4th‑order IFCs (subcommands: nep/dp/ploymp/mtp), uses na nb nc
- scph: run self‑consistent phonon workflow (subcommands: nep/dp/hiphive/ploymp/mtp)
- plot_phband: plot phonon band structure from FORCE_CONSTANTS
  - Single colorbar on the right; each dataset has a tick label; `--labels` optional

Run `fcsorder --help` or `fcsorder <subapp> --help` for full details.

## MTP Notes

- Requires `mlp` binary accessible via `PATH` or specify `--mtp-exe`.
- Temporary files are written under the system temp directory by default.
- In calculations, unique chemical symbols are collected from the ASE Atoms object automatically.

## Development

- Run formatting and basic lint if desired
- Contribute via PRs on GitHub: https://github.com/gtiders/fcs-order


## License

GPL-3.0-or-later
