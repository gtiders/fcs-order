# FCS-Order: Force Constants Calculation Suite

Repository: [https://github.com/gtiders/fcs-order](https://github.com/gtiders/fcs-order)

FCS-Order is a comprehensive Python package for calculating third-order and fourth-order force constants, with support for machine learning potentials and integration with the ALAMODE software package. This project modernizes the original thirdorder/fourthorder tools by converting them to Python 3+ and adding advanced features.

## Features

### 🎯 Core Capabilities

- **Third-order force constants** calculation using finite displacement method
- **Fourth-order force constants** calculation using finite displacement method  
- **Machine learning potential integration** for direct force constant calculation
- **ALAMODE integration** via DFTSETS file generation
- **Effective harmonic parameter fitting** at finite temperatures

### 🔬 Supported Calculators

- **NEP** (Neuroevolution Potential)
- **DeepMD** (Deep Potential)
- **HiPhive** (Force Constant Potential)
- **Polymlp** (Machine Learning Potential)

### 📦 Key Modules

#### 1. `thirdorder` - Third-order Force Constants

Traditional finite displacement method for 3-phonon calculations:

```bash
# Generate displacement structures
fcsorder thirdorder sow 2 2 2 --cutoff -2

# Extract force constants from VASP calculations  
fcsorder thirdorder reap  2 2 2  vasprun1.xml vasprun2.xml ...  --cutoff -2

# Direct calculation using ML potential (alternative approach)
fcsorder thirdorder get-fc 2 2 2 --cutoff -2 --calc nep --potential nep.txt
```

#### 2. `fourthorder` - Fourth-order Force Constants

Traditional finite displacement method for 4-phonon calculations:

```bash
# Generate displacement structures
fcsorder fourthorder sow 2 2 2 --cutoff -2

# Extract force constants from VASP calculations
fcsorder fourthorder reap 2 2 2 --cutoff -2 vasprun1.xml vasprun2.xml ...

# Direct calculation using ML potential (alternative approach)
fcsorder fourthorder get-fc 2 2 2 --cutoff -2 --calc nep --potential nep.txt
```

#### 3. `generate2alm` - ALAMODE DFTSETS Generation

Generate training data for ALAMODE force constant fitting:

```bash
# Extract data from completed VASP calculations
fcsorder generate2alm SPOSCAR disp1.xml disp2.xml --output DFSETs

# Generate data using ML potential
fcsorder generate2alm SPOSCAR disp1.xml disp2.xml --calc nep --potential nep.txt --run_with_potential --output DFSETs

# Sample every 5th configuration
fcsorder generate2alm SPOSCAR disp1.xml disp2.xml --delta 5 --output DFSETs

# Correct forces by subtracting reference forces
fcsorder generate2alm SPOSCAR.xml disp1.xml disp2.xml --correct_force --output DFSETs
```

#### 4. `effective-harmonic` - Finite Temperature Properties

Fit effective harmonic parameters at specified temperatures:

```bash
# Run MD at multiple temperatures using NEP potential
fcsorder effective-harmonic 2 2 2 --calc nep --potential nep.txt \
  --temperatures "2000,1000,300" --outdir md_runs

# Use different supercell size for training vs reference
fcsorder effective-harmonic 3 3 3 --prim POSCAR --sposcar SPOSCAR \
  --calc dp --potential model.pb --temperatures "500,300"

# Custom MD parameters
fcsorder effective-harmonic 2 2 2 --calc hiphive --potential potential.fcp \
  --temperatures "800" --neq 5000 --nprod 10000 --dt 2.0 --dump 50
```

## Installation

### Basic Installation

```bash
pip install fcs-order
```

### With Optional Dependencies

```bash
# For DeepMD support
pip install "fcs-order[deepmd]"

# For HiPhive support  
pip install "fcs-order[hiphive]"

# For Calorine (NEP) support
pip install "fcs-order[calorine]"

# For Polymlp support
pip install "fcs-order[pypolymlp]"

# Install all optional dependencies
pip install "fcs-order[all]"
```

## Requirements

- Python 3.9+
- ASE (Atomic Simulation Environment)
- NumPy, SciPy
- Click for CLI interface
- spglib for symmetry analysis

## License

This project builds upon the original thirdorder/fourthorder tools and maintains compatibility while adding modern features.

## Citation

If you use this software in your research, please cite both the original thirdorder/fourthorder work and acknowledge the FCS-Order enhancements for machine learning potential integration and ALAMODE compatibility.
