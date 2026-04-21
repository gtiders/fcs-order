# MLFCS

<p align="center">
  <strong>Machine Learning Force Constant Suite</strong><br/>
  A practical toolkit for 2nd/3rd/4th-order force constants with CLI and Python APIs.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-GPLv3-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue.svg">
  <img alt="Version" src="https://img.shields.io/badge/version-2.0.0-0A7EA4.svg">
</p>

<p align="center">
  <a href="README_ZH.md">中文文档</a> ·
  <a href="docs/quickstart.md">Quickstart</a>
</p>

## What MLFCS Is

MLFCS is a refactored, workflow-oriented force-constant suite built around three layers:

- `thirdorder` / `fourthorder` command-line workflows (`sow` + `reap`), compatible with classic high-order finite-difference usage patterns.
- `secondorder` Python APIs (`MLPHONON`, `MLPSSCHA`) for harmonic and SSCHA calculations using ASE calculators.
- `hifinit` single-run API (`HifinitRun`) that computes force constants by finite differences first, then projects them into hiPhive parameter space; this helps avoid cross-order contamination often seen in pure global fitting workflows.

The project focuses on reliable interfaces, scriptability, and practical integration with modern ASE-based model potentials.

## Why Use MLFCS

- One codebase for 2nd, 3rd, and 4th-order force constants.
- CLI mode for file-based DFT/force workflows.
- Python mode for direct ASE-calculator execution.
- Explicit phonopy interface control for structure and force parsing.
- Output formats compatible with phonopy/phono3py/ShengBTE-style downstream tools.

## HIFINIT Rationale

`HifinitRun` is designed as a finite-difference-first workflow, not a pure global fitting workflow:

1. Build force-constant tensors from finite differences on orbit prototypes.
2. Project the result into hiPhive parameter space to enforce symmetry/ASR constraints.

Why this matters:
- Cleaner order separation: each order is built from its corresponding finite-difference evaluations.
- Reduced cross-order error leakage: mitigates the common issue where fitting redistributes errors across different orders.
- Keeps hiPhive benefits: symmetry handling, parameter organization, and practical output interoperability.

### Compared to Pure Fitting Workflows

| Aspect | HIFINIT (MLFCS) | Pure fitting workflow (common hiPhive usage) |
|---|---|---|
| FC source | Finite-difference tensors first, then projection to parameter space | Global parameter fitting from force datasets |
| ASR/symmetry constraints | Explicitly enforced in hiPhive parameter space; ASR and symmetry constraints are satisfied within numerical precision (including translation/rotation-related constraints) | Depends more on fitting setup and data quality; can show drift or cross-order error redistribution |
| Order separability | Stronger: low-/high-order contributions are more traceable | Weaker: order mixing/error absorption is more likely |
| Computational cost | High, especially for higher order and larger supercells | Usually lower (depends on dataset/model size) |
| Recommended calculator | Strongly recommended to use ASE ML potentials (NEP/MACE/DP/GAP, etc.) | DFT or ML potentials, depending on target |

Practical guidance:
- For 3rd/4th-order or large systems, `HifinitRun` is typically practical only with ML potentials.
- If your priority is tighter ASR/symmetry-consistent force constants, HIFINIT is usually the better route.

## Project Scope (Important)

- The phonopy interface layer in MLFCS is an extension for structure I/O and force parsing.
- It does **not** replace the core third-/fourth-order reconstruction logic.
- Supported interfaces depend on your installed phonopy version. Discover them with:

```bash
thirdorder interfaces
fourthorder interfaces
```

## Installation

```bash
git clone https://github.com/gtiders/mlfcs.git
cd mlfcs
pip install .
```

Requirements:
- Python `>= 3.12`
- Dependencies are defined in [`pyproject.toml`](pyproject.toml)

## Fast Start

### 1) Third-order CLI

```bash
# Generate displaced supercells
thirdorder sow 4 4 4 --cutoff -3 --interface vasp --format vasp

# Run external force calculations for each generated structure

# Reconstruct FORCE_CONSTANTS_3RD
thirdorder reap 4 4 4 --cutoff -3 \
  --interface vasp \
  --forces-interface vasp \
  --forces "./3RD_runs/*/vasprun.xml"
```

### 2) Fourth-order CLI

```bash
fourthorder sow 3 3 3 --cutoff -2 --interface vasp --format vasp

fourthorder reap 3 3 3 --cutoff -2 \
  --interface vasp \
  --forces-interface vasp \
  --forces "./4TH_runs/*/vasprun.xml"
```

### 3) Python API: `ThirdOrderRun` (3rd order, ASE calculator)

```python
from mlfcs.thirdorder import ThirdOrderRun
from calorine.calculators import CPUNEP

calc = CPUNEP("nep.txt")
runner = ThirdOrderRun(na=3, nb=3, nc=3, cutoff=-3, structure_file="POSCAR")
runner.run_calculator(calc)  # writes FORCE_CONSTANTS_3RD
```

### 4) Python API: `FourthOrderRun` (4th order, ASE calculator)

```python
from mlfcs.fourthorder import FourthOrderRun
from calorine.calculators import CPUNEP

calc = CPUNEP("nep.txt")
runner = FourthOrderRun(na=3, nb=3, nc=3, cutoff=-2, structure_file="POSCAR")
runner.run_calculator(calc)  # writes FORCE_CONSTANTS_4TH
```

### 5) Python API: `MLPHONON` (2nd order)

```python
from ase.io import read
from calorine.calculators import CPUNEP
from mlfcs.secondorder import MLPHONON

prim = read("POSCAR")
calc = CPUNEP("nep.txt")

phonon = MLPHONON(
    structure=prim,
    calculator=calc,
    supercell_matrix=[2, 2, 2],
    kwargs_generate_displacements={"distance": 0.01},
)
phonon.run()
phonon.write("FORCE_CONSTANTS")  # text
phonon.write("fc2.hdf5")         # hdf5
```

### 6) Python API: `MLPSSCHA`

```python
from ase.io import read
from calorine.calculators import CPUNEP
from mlfcs.secondorder import MLPSSCHA

prim = read("POSCAR")
calc = CPUNEP("nep.txt")

sscha = MLPSSCHA(
    unitcell=prim,
    calculator=calc,
    supercell_matrix=[3, 3, 3],
    temperature=300,
    number_of_snapshots=1000,
    max_iterations=20,
    avg_n_last_steps=5,
    fc_output="fc2_sscha.hdf5",
    fc_output_format="hdf5",
)
sscha.run()
```

### 7) Python API: `HifinitRun`

```python
from ase.io import read
from calorine.calculators import CPUNEP
from mlfcs.hifinit import HifinitRun

prim = read("POSCAR")
supercell = read("SPOSCAR")
calc = CPUNEP("nep.txt")

runner = HifinitRun(
    primitive=prim,
    supercell=supercell,
    calculator=calc,
    displacement=0.005,
    cutoffs=[None, None, 4.0],
)
runner.run(out_dir="./hifinit_results", verbose=True)
```

## CLI Reference (`thirdorder` / `fourthorder`)

Both tools share the same command shape:

```bash
<tool> {sow|reap|interfaces} [na nb nc] [options]
```

Common options:

| Option | Applies to | Meaning |
|---|---|---|
| `na nb nc` | `sow`, `reap` | Supercell multipliers along `a`, `b`, `c` |
| `--cutoff` | `sow`, `reap` | Positive number: distance cutoff; negative integer: neighbor shell (e.g. `-3`) |
| `-i`, `--input` | `sow`, `reap` | Input structure path (default `POSCAR`) |
| `--interface` | `sow`, `reap` | Structure I/O interface name |
| `--forces-interface` | `reap` | Force parser interface (default: same as `--interface`) |
| `--hstep` | `sow`, `reap` | Displacement step in nm |
| `--symprec` | `sow`, `reap` | Symmetry tolerance |
| `-f`, `--format` | `sow` | `vasp` or `same` (`same` writes using `--interface`) |
| `--forces` | `reap` | Force files or glob patterns |

Notes:
- `reap` requires exactly the expected number of force sets.
- `.xyz` / `.extxyz` trajectories are intentionally rejected in CLI `reap`.

## Output Files

### `thirdorder` / `fourthorder`
- `FORCE_CONSTANTS_3RD`
- `FORCE_CONSTANTS_4TH`

### `MLPHONON` / `MLPSSCHA`
- Text: `FORCE_CONSTANTS`
- HDF5: `*.hdf5`

### `HifinitRun` (`out_dir`)
- `potential.fcp`
- `FORCE_CONSTANTS_2ND`, `fc2.hdf5`
- `FORCE_CONSTANTS_3RD`, `fc3.hdf5` (if order >= 3)
- `FORCE_CONSTANTS_4TH` (if order >= 4)

## Best Practices

- Keep `--interface` and `--forces-interface` explicit in scripts.
- For API workflows, validate calculator reproducibility on a small supercell first.
- In manual ASE loops, freeze computed results with `SinglePointCalculator` before writing trajectories to avoid unintended recalculation/caching side effects.

## FAQ

### Is this a drop-in replacement for legacy `thirdorder.py` / `fourthorder.py`?
It is a refactored implementation with compatible workflow concepts (`sow`/`reap`), plus explicit interface control and Python APIs.

### Which DFT codes are supported?
MLFCS uses phonopy interface names for structure/force I/O. The concrete list depends on your installed phonopy build; use `thirdorder interfaces` to inspect what is available in your environment.

### Can I use machine-learning potentials?
Yes. Python APIs are designed to work with ASE calculators (examples use `calorine` `CPUNEP`, but the interface is ASE-calculator based).

## Citation and Acknowledgements

MLFCS is developed on top of ideas and workflows from:
- `thirdorder.py` (ShengBTE ecosystem)
- `fourthorder`
- `phonopy`
- `hiPhive`

Thanks to all original authors and maintainers.

## License

This project is distributed under **GNU General Public License v3.0**. See [`LICENSE`](LICENSE).
