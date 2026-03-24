# MLFCS (Machine Learning Force Constant Suite)

![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

[中文说明 (Chinese Version)](README_ZH.md)

MLFCS is a modern suite for calculating Anharmonic Force Constants, designed to provide efficient and easy-to-use solutions for high-throughput materials calculation.

This project is a deep refactoring and optimization based on the classic `thirdorder.py` and `fourthorder.py`.

## 🤖 AI-Powered Documentation

**New to this project?** You can:

*   📖 **Feed this README to an AI assistant** (e.g., ChatGPT, Claude, DeepSeek) to quickly understand the usage and get started.
*   💻 **Feed the entire codebase to an AI assistant** for in-depth understanding of implementation details and advanced usage.

We also welcome community contributions:
*   🐛 **Report bugs or request features**: [Open an Issue](https://github.com/gtiders/mlfcs/issues)
*   🔧 **Submit improvements**: [Pull Requests](https://github.com/gtiders/mlfcs/pulls) are always welcome!
*   📧 **Contact via email**: gtiders@qq.com

## ⚠️ Version Notice

The `main` branch contains ongoing development and experimental features (e.g., C++ `unordered_map` optimization). For production use, stick to the [releases](https://github.com/gtiders/mlfcs/releases).

## 📋 Output Format

MLFCS outputs force constants in its native format. **It does not provide built-in support for phono3py format.** If you need phono3py-compatible output, you can use [hiPhive](https://hiphive.materialsmodeling.org/) for format conversion. Example:

```python
from hiphive import ForceConstants

# Read MLFCS output and convert to phono3py format
# See hiphive documentation for details
```

## ✨ Key Features

*   **Pure Python**: Completely removed the dependency on `syplib` C extensions, solving tedious compilation and dependency issues.
*   **Extreme Performance**:
    *   🚀 **5x Faster**: Significant speed improvements through algorithm optimization.
    *   💾 **1% Memory Usage**: Optimized memory management to easily handle large supercell systems.
    *   📦 **Easy Installation**: Supports standard `pip` installation, works out of the box.
*   **Comprehensive Features**: Supports generation (Sow) and extraction (Reap) of Third-order and Fourth-order force constants.
*   **Multi-format Support**: Compatible with **VASP** and **XYZ/ExtXYZ** formats, facilitating integration with various calculation codes (e.g., ASE calculators).

## 🛠️ Installation

You can install this project directly via pip:

```bash
git clone https://github.com/gtiders/mlfcs.git
cd mlfcs
pip install .
```

### ⚠️ For Legacy Systems (CentOS 7, etc.)

On older systems with outdated compilers (GCC < 9), NumPy 2.0+ may cause compilation issues. Before installing, modify `pyproject.toml`:

```diff
- requires = ["setuptools>=80.0.0", "wheel", "cython>=3.0.0", "numpy>=2.0.0"]
+ requires = ["setuptools>=80.0.0", "wheel", "cython>=3.0.0", "numpy<2.0.0"]
```

Then install:

```bash
pip install .
```

## 📖 Usage Guide

The suite includes two main commands: `thirdorder` and `fourthorder`. Each command contains `sow` (generate displacements) and `reap` (collect forces and calculate force constants) subcommands.

### Interface Selection (Required for non-VASP inputs)

Structure reading now uses an explicit `--interface` option:

- Default is `--interface vasp`.
- For ABACUS `STRU`, use `--interface abacus`.
- If parsing fails, MLFCS prints supported interfaces (from phonopy), e.g. `abacus`, `vasp`, `qe`, `cp2k`, `aims`.

### CLI Parameter Reference (`thirdorder` / `fourthorder`)

| Parameter | Required | Scope | Description |
| --- | --- | --- | --- |
| `command` | Yes | all | Subcommand: `sow` or `reap`. |
| `na nb nc` | Yes | all | Supercell size along `a/b/c` directions, e.g. `4 4 4`. |
| `--cutoff` | Yes | all | Cutoff rule. Negative integer means neighbor shell index (e.g. `-3` = 3rd neighbor). Positive number means distance cutoff. |
| `-i, --input` | No | all | Input structure file. Default: `POSCAR`. |
| `--interface` | No | all | Structure parsing interface. Default: `vasp`. Use explicit values matching your input, e.g. `abacus`, `qe`, `cp2k`, `aims`. |
| `--symprec` | No | all | Symmetry precision. Default from code constants (currently `1e-5`). |
| `--hstep` | No | all | Displacement step size. Default from code constants (currently `0.001`). |
| `-f, --format` | No | sow | Displacement output format: `vasp` (multiple `*.POSCAR.*` files) or `xyz` (single displacement trajectory file). Default: `vasp`. |
| `--forces` | Yes for `reap` | reap | One or more force-output files/patterns (glob is supported). **Not supported in CLI:** `xyz/extxyz` force trajectories. |
| `--forces-interface` | No | reap | Interface used to parse force files. Default: reuse `--interface`. |

### Minimal End-to-End Workflow (Sow -> Calculate -> Reap)

```bash
# 1) Generate displacements
thirdorder sow 4 4 4 --cutoff -3 --format vasp

# 2) Run external force calculations for each displacement
# Recommended layout: one folder per displacement index from sow output
# e.g. 3RD_runs/0001/vasprun.xml, 3RD_runs/0002/vasprun.xml, ...

# 3) Reap force constants (sorted folder order -> displacement ID order)
thirdorder reap 4 4 4 --cutoff -3 \
  --forces $(find ./3RD_runs -name "vasprun.xml" | sort -V) \
  --forces-interface vasp
```

### Third-order Force Constants (Thirdorder)

#### 1. Generate Displacements (Sow)

Generate supercell displacement structures for third-order force constant calculations.

```bash
# Basic usage: 4x4x4 supercell, cutoff 3rd neighbor (negative for neighbor index)
thirdorder sow 4 4 4 --cutoff -3

# ABACUS input (STRU)
thirdorder sow 4 4 4 --cutoff -3 -i STRU --interface abacus

# QE input example
thirdorder sow 4 4 4 --cutoff -3 -i qe.in --interface qe

# CP2K input example
thirdorder sow 4 4 4 --cutoff -3 -i cp2k.inp --interface cp2k

# Specify cutoff radius as 5.0 nm (positive for distance)
thirdorder sow 4 4 4 --cutoff 5.0

# Output in xyz format (Recommended for ML Potentials)
thirdorder sow 4 4 4 --cutoff -3 --format xyz

# Custom parameters: displacement step 0.001, symmetry precision 1e-4
thirdorder sow 4 4 4 --cutoff -3 --hstep 0.001 --symprec 1e-4
```

This will generate a series of `3RD.POSCAR.*` files (or a single `3RD.displacements.xyz` file).

#### 2. Calculate Forces (External Step)

Run force calculations for every displaced structure generated by `sow`:
- If `sow --format vasp`: run your DFT/calculator jobs on `3RD.POSCAR.*`.
- If `sow --format xyz`: run calculator jobs from `3RD.displacements.xyz` (commonly used in Python workflows).

**Key Point**: CLI `reap` expects calculator output files parsed by phonopy interfaces; keep file naming/order deterministic.

#### 3. Collect Force Constants (Reap)

Use the `reap` command to extract force constants from the calculated files.

```bash
# Basic usage: Extract from VASP xml/OUTCAR files
thirdorder reap 4 4 4 --cutoff -3 --forces vasprun.xml.* --forces-interface vasp

# ABACUS force outputs
thirdorder reap 4 4 4 --cutoff -3 --forces running_scf.log.* --forces-interface abacus
```

The results will be output to the `FORCE_CONSTANTS_3RD` file.

Note:
- CLI `reap` now parses force files via phonopy interfaces and does not accept `xyz/extxyz`.
- For `xyz` force trajectories, use the Python library workflow instead.

##### Reap Multi-file Ordering

`reap` maps force files by sorted file order to displacement IDs (`1..N`), so ordering must be deterministic.
Recommended: keep one directory per displacement ID (same numbering as `sow` output), e.g. `0001/`, `0002/`, ...

```bash
# VASP: simple wildcard (works when file names are zero-padded or naturally ordered)
thirdorder reap 4 4 4 --cutoff -3 --forces vasprun.xml.* --forces-interface vasp

# ABACUS: explicit find + version sort
thirdorder reap 4 4 4 --cutoff -3 \
  --forces $(find ./abacus_runs -name "running_scf.log.*" | sort -V) \
  --forces-interface abacus

# QE: collect pw.x outputs
thirdorder reap 4 4 4 --cutoff -3 \
  --forces $(find ./qe_runs -name "pw.out.*" | sort -V) \
  --forces-interface qe

# CP2K: collect output logs
thirdorder reap 4 4 4 --cutoff -3 \
  --forces $(find ./cp2k_runs -name "*.out" | sort -V) \
  --forces-interface cp2k
```

If in doubt, print and verify ordering first:
```bash
find ./abacus_runs -name "running_scf.log.*" | sort -V
```

### Fourth-order Force Constants (Fourthorder)

The workflow is similar to thirdorder.

#### 1. Generate Displacements (Sow)

```bash
# Generate 3x3x3 supercell, 2nd neighbor cutoff
fourthorder sow 3 3 3 --cutoff -2

# ABACUS input (STRU)
fourthorder sow 3 3 3 --cutoff -2 -i STRU --interface abacus

# QE input example
fourthorder sow 3 3 3 --cutoff -2 -i qe.in --interface qe

# CP2K input example
fourthorder sow 3 3 3 --cutoff -2 -i cp2k.inp --interface cp2k
```

This will generate `4TH.POSCAR.*` files.

#### 2. Collect Force Constants (Reap)

```bash
fourthorder reap 3 3 3 --cutoff -2 --forces vasprun.xml.* --forces-interface vasp

# ABACUS force outputs
fourthorder reap 3 3 3 --cutoff -2 --forces running_scf.log.* --forces-interface abacus

# QE force outputs (same ordering rule as thirdorder)
fourthorder reap 3 3 3 --cutoff -2 \
  --forces $(find ./qe_runs -name "pw.out.*" | sort -V) \
  --forces-interface qe
```

The results will be output to the `FORCE_CONSTANTS_4TH` file.

## 🐍 Python API Usage (Advanced)

Besides the CLI tools, you can call the core classes directly in Python scripts. This is very convenient for integrating ASE calculators (e.g., NEP, GAP, MACE, DP, etc.) without intermediate file I/O.

### Basic Example

```python
from mlfcs.thirdorder import ThirdOrderRun
# Assuming you use calorine's CPUNEP calculator, or any ASE Calculator
from calorine.calculators import CPUNEP

# Initialize runner
# kwargs: na=4, nb=4, nc=4, cutoff=-3 (3rd neighbor)
runner = ThirdOrderRun(4, 4, 4, -3)

# Define ASE calculator
calc = CPUNEP("nep.txt")

# Run calculation directly, no manual file I/O needed
runner.run_calculator(calc)
```

### Parameter Overrides (H & Symprec)

You can customize the displacement step (`h`) and symmetry precision (`symprec`) during initialization:

```python
# h: displacement step (default usually 0.04 or similar, depends on order)
# symprec: symmetry precision (default 1e-5)
runner = ThirdOrderRun(4, 4, 4, -3, h=0.001, symprec=1e-4)
```

### Harmonic Phonon Calculation (MLPHONON)

You can use the `MLPHONON` class to calculate harmonic force constants using any ASE calculator.

```python
from mlfcs.phonon import MLPHONON
from ase.io import read
from calorine.calculators import CPUNEP

# Read structure
structure = read("POSCAR")

# Initialize calculator
calc = CPUNEP("nep.txt")

# Setup phonon calculation
phonon = MLPHONON(
    structure=structure,
    calculator=calc,
    supercell_matrix=[2, 2, 2],  # Supercell expansion
    kwargs_generate_displacements={"distance": 0.01}  # Optional
)

# Run calculation
phonon.run()

# Write force constants to file
phonon.write("FORCE_CONSTANTS")

# Access Phonopy object for further analysis
phonon.phonopy.run_mesh([20, 20, 20])
phonon.phonopy.run_total_dos()
```

### Self-Consistent Harmonic Approximation (SSCHA)

You can use the `MLPSSCHA` class to perform SSCHA calculations using any ASE calculator (e.g., NEP).

```python
from mlfcs.sscha import MLPSSCHA
from calorine.calculators import CPUNEP

# Initialize calculator
calc = CPUNEP("nep.txt")

# Setup SSCHA run
sscha = MLPSSCHA(
    unitcell="./POSCAR",         # Path to primitive cell
    supercell_matrix=[3, 3, 3],  # Supercell expansion
    calculator=calc,             # ASE Calculator
    temperature=300,             # Temperature in K
    number_of_snapshots=1000,    # Structures per iteration
    max_iterations=20,           # Max iterations
    avg_n_last_steps=5,          # Average last 5 steps for final result
    fc_output="FORCE_CONSTANTS"  # Output filename
)

# Run calculation
sscha.run()
```

### Best Practice: Preventing Calculator Caching Issues

If you choose to manually loop through structures for calculation (instead of using `runner.run_calculator`), be mindful of the ASE calculator's caching mechanism. To prevent `write` operations from accidentally triggering re-calculation or writing old data, it is recommended to "freeze" results using `SinglePointCalculator`.

```python
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

# ... inside a manual loop ...
atoms.calc = calc  # Attach your main calculator (NEP, VASP, etc.)
forces = atoms.get_forces()
energy = atoms.get_potential_energy()

# [Critical Step] Detach main calculator, store static results in SinglePointCalculator
# This allows safe writing to file, avoiding re-triggering calc or mixing frame data
atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)

# Now safe to write
write("forces.xyz", atoms, format="extxyz", append=True)
```

## 🙏 Acknowledgments

The development of this project relies on contributions from the open-source community. Special thanks to the following pioneering projects:

*   **[ShengBTE / thirdorder.py](https://www.shengbte.org/announcements/thirdorderpyv110released)**: Thanks to Wu Li et al. for the original `thirdorder.py`, laying the foundation for anharmonic phonon calculations.
*   **[Fourthorder](https://github.com/FourPhonon/Fourthorder)**: Thanks to Han, Zherui et al. for the fourth-order force constant calculation code.

Based on these excellent works, we have focused on improving software engineering architecture, installation experience, and execution efficiency, hoping to provide better tools for the community.

## 📄 License

This project is licensed under the GNU General Public License v3.0 (GPLv3).
