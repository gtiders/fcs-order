# MLFCS (Machine Learning Force Constant Suite)

![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

[中文说明 (Chinese Version)](README_ZH.md)

MLFCS is a modern suite for calculating Anharmonic Force Constants, designed to provide efficient and easy-to-use solutions for high-throughput materials calculation.

This project is a deep refactoring and optimization based on the classic `thirdorder.py` and `fourthorder.py`.

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

## 📖 Usage Guide

The suite includes two main commands: `thirdorder` and `fourthorder`. Each command contains `sow` (generate displacements) and `reap` (collect forces and calculate force constants) subcommands.

### Third-order Force Constants (Thirdorder)

#### 1. Generate Displacements (Sow)

Generate supercell displacement structures for third-order force constant calculations.

```bash
# Basic usage: 4x4x4 supercell, cutoff 3rd neighbor (negative for neighbor index)
thirdorder sow 4 4 4 -3

# Specify cutoff radius as 5.0 nm (positive for distance)
thirdorder sow 4 4 4 5.0

# Output in xyz format (Recommended for ML Potentials)
thirdorder sow 4 4 4 -3 --format xyz

# Custom parameters: displacement step 0.001, symmetry precision 1e-4
thirdorder sow 4 4 4 -3 --hstep 0.001 --symprec 1e-4
```

This will generate a series of `3RD.POSCAR.*` files (or a single `3RD.displacements.xyz` file).

#### 2. Calculate Forces (External Step)

If you used the `xyz` format, you need to calculate the forces for all structures in `3RD.displacements.xyz` using your own calculator (e.g., VASP, LAMMPS, or ML Potentials).

**Key Point**: Ensure the calculated file includes force information and preserves the structure order (or includes the `config_id` attribute).

#### 3. Collect Force Constants (Reap)

Use the `reap` command to extract force constants from the calculated files.

```bash
# Basic usage: Extract from VASP xml/OUTCAR files
thirdorder reap 4 4 4 -3 --forces vasprun.xml.*

# Advanced usage: Extract from calculated XYZ file (e.g., calculated_forces.xyz)
# Note: If a non-default hstep was used during sow, it must be specified here
thirdorder reap 4 4 4 -3 --forces calculated_forces.xyz --hstep 0.001
```

The results will be output to the `FORCE_CONSTANTS_3RD` file.

### Fourth-order Force Constants (Fourthorder)

The workflow is similar to thirdorder.

#### 1. Generate Displacements (Sow)

```bash
# Generate 3x3x3 supercell, 2nd neighbor cutoff
fourthorder sow 3 3 3 -2
```

This will generate `4TH.POSCAR.*` files.

#### 2. Collect Force Constants (Reap)

```bash
fourthorder reap 3 3 3 -2 --forces vasprun.xml.*
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
