# [中文文档 | Chinese README](README_zh.md)

# fcs-order

fcs-order is a toolkit for calculating lattice thermal conductivity and phonon properties, supporting 2nd, 3rd, and 4th-order force constants (FCs). It integrates various Machine Learning Potentials (MLP) and Self-Consistent Phonon (SCPH) calculations.

> **Note**: The CLI has been unified into a single `fcsorder` command. Old workflow commands like `sow`/`reap` are replaced by unified `fcX` commands that handle displacement generation, force calculation (via MLP), and reconstruction in one step.

## Installation

```bash
pip install fcs-order
```

**Backends (Install as needed):**
- **NEP**: `pip install calorine`
- **DeepMD**: `pip install deepmd-kit`
- **PolyMLP**: `pip install pypolymlp`
- **Hiphive**: `pip install hiphive`
- **TACE**: `pip install tace`
- **Phonopy**: `pip install phonopy` (Required for fc2/scph)
- **MACE** reference the officical document of mace
## CLI Usage

The main command is `fcsorder` (or `forcekit`).

```bash
fcsorder --help
```

### 1. Second-Order Force Constants (`fc2`)

Calculate harmonic force constants using Phonopy and an ML potential.

```bash
fcsorder fc2 [NA] [NB] [NC] -c <calculator> -p <potential> [options]
```

**Arguments:**
- `NA`, `NB`, `NC`: Supercell repetition (e.g., `2 2 2`).
- `-c, --calculator`: Backend type (`nep`, `dp`, `tace`, `hiphive`, etc.).
- `-p, --potential`: Path to the potential model file.
- `-s, --structure`: Input structure file (default: `POSCAR`).
- `-o, --output`: Output filename (default: `FORCE_CONSTANTS`).
- `-f, --output-format`: `text` (Phonopy format) or `hdf5`.

**Example:**
```bash
fcsorder fc2 2 2 2 -c nep -p nep.txt
```

### 2. Third-Order Force Constants (`fc3`)

Calculate cubic anharmonic force constants.

```bash
fcsorder fc3 [NA] [NB] [NC] -c <calculator> -p <potential> -k <cutoff> [options]
```

**Arguments:**
- `-k, --cutoff`: Cutoff radius. Negative (e.g., `-3`) for nearest neighbors, positive (e.g., `0.5`) for nm.
- `-f, --output-format`: `text` (default, writes `FORCE_CONSTANTS_3RD`) or `hdf5` (writes `fc3.hdf5` via Hiphive).
- `-w, --save-intermediate`: Save displaced structures/forces.
- `--device`: `cpu` or `cuda`.

**Example:**
```bash
# Calculate and export to Phono3py HDF5 format
fcsorder fc3 2 2 2 -c nep -p nep.txt -k -3 -f hdf5
```

### 3. Fourth-Order Force Constants (`fc4`)

Calculate quartic anharmonic force constants.

```bash
fcsorder fc4 [NA] [NB] [NC] -c <calculator> -p <potential> -k <cutoff> [options]
```

**Arguments:**
- Similar to `fc3`.
- **Note**: Only text output (`FORCE_CONSTANTS_4TH`) is supported.

**Example:**
```bash
fcsorder fc4 2 2 2 -c dp -p full.pb -k -2
```

### 4. Self-Consistent Phonons (`scph`)

Run non-perturbative phonon renormalization using SCPH.

```bash
fcsorder scph [NA] [NB] [NC] -c <calculator> -p <potential> -T <temperatures> -k <cutoff> [options]
```

**Arguments:**
- `-T, --temperatures`: Comma-separated temperatures (e.g., `100,300`).
- `-k, --cutoff`: Cluster space cutoff (nm).
- `-a, --alpha`: Mixing parameter (default: 0.2).
- `-i, --num-iterations`: Max iterations (default: 30).
- `-n, --num-structures`: Structures per iteration (default: 500).
- `-f, --output-format`: Output format for effective FCs (`text` or `hdf5`).

**Example:**
```bash
fcsorder scph 2 2 2 -c nep -p nep.txt -T 300 -k 4.5
```

## Structure Generation Utilities

### Phonon Rattle (`phonon-rattle`)
Generate thermal configurations based on harmonic phonons.
```bash
fcsorder phonon-rattle POSCAR --force-constants-file FORCE_CONSTANTS -T 300 -n 10
```

### Monte Carlo Rattle (`monte-rattle`)
Generate configurations using MC rattle to avoid close contacts.
```bash
fcsorder monte-rattle POSCAR -n 10 --d-min 1.0 --rattle-amplitude 0.05
```

### Simple Rattle (`rattle`)
Generate configurations using uncorrelated Gaussian noise.
```bash
fcsorder rattle POSCAR -n 10 --rattle-amplitude 0.05
```

## License
Apache-2.0
