# FCS-Order: Third and Fourth Order Force Constants for VASP

A Python package for calculating third-order and fourth-order force constants in crystalline materials using VASP density functional theory calculations.

## Overview

FCS-Order provides tools to:
- Generate displaced supercell configurations ("sow" phase)
- Process VASP output to extract force constants ("reap" phase)
- Calculate third-order and fourth-order interatomic force constants (IFCs)
- Support both third-order and fourth-order calculations in a unified framework

## Features

- **Third-order force constants**: Calculate cubic anharmonic terms
- **Fourth-order force constants**: Calculate quartic anharmonic terms  
- **VASP integration**: Seamless workflow with VASP DFT calculations
- **Symmetry analysis**: Automatic symmetry reduction to minimize computational cost
- **Python 3 compatibility**: Modern Python implementation with Cython acceleration
- **Command-line interface**: Simple CLI tools for both third and fourth order calculations

## Installation

### Prerequisites

- Python 3.8+
- VASP (for DFT calculations)
- spglib (for symmetry analysis)

### Install from source

```bash
git clone <repository-url>
cd fcs-order
uv pip install -e .
```

### Verify installation

```bash
# Check if commands are available
thirdorder --help
fourthorder --help
```

## Usage

### Third-order force constants

#### 1. Generate displaced configurations (sow phase)

```bash
thirdorder sow NA NB NC -c CUTOFF
```

Where:
- `NA`, `NB`, `NC`: Supercell dimensions
- `-c CUTOFF`: Cutoff value (negative for nearest neighbors, positive for distance in nm)

This generates:
- `3RD.SPOSCAR`: Undisplaced supercell coordinates
- `3RD.POSCAR.XXXXX`: Displaced configurations for VASP calculations

#### 2. Process VASP results (reap phase)

```bash
thirdorder reap NA NB NC vasprun1.xml vasprun2.xml ... -c CUTOFF
```

This processes the VASP output files and generates the third-order force constants.

### Fourth-order force constants

#### 1. Generate displaced configurations (sow phase)

```bash
fourthorder sow NA NB NC -c CUTOFF
```

This generates:
- `4TH.SPOSCAR`: Undisplaced supercell coordinates  
- `4TH.POSCAR.XXXXX`: Displaced configurations for VASP calculations

#### 2. Process VASP results (reap phase)

```bash
fourthorder reap NA NB NC vasprun1.xml vasprun2.xml ... -c CUTOFF
```

This processes the VASP output files and generates the fourth-order force constants.

## Workflow Example

### Third-order calculation

1. **Prepare input**: Ensure `POSCAR` file is present in your working directory
2. **Generate configurations**: `thirdorder sow 2 2 2 -c -1`
3. **Run VASP**: Calculate forces for each `3RD.POSCAR.XXXXX` configuration
4. **Extract force constants**: `thirdorder reap 2 2 2 *.xml -c -1`

### Fourth-order calculation

1. **Prepare input**: Ensure `POSCAR` file is present in your working directory
2. **Generate configurations**: `fourthorder sow 2 2 2 -c -1`
3. **Run VASP**: Calculate forces for each `4TH.POSCAR.XXXXX` configuration  
4. **Extract force constants**: `fourthorder reap 2 2 2 *.xml -c -1`

## Input Files

- **POSCAR**: VASP structure file (required in working directory)
- **vasprun.xml files**: VASP output files from force calculations

## Output Files

### Third-order
- `FORCE_CONSTANTS_3RD`: Third-order force constants
- `3RD.SPOSCAR`: Supercell structure
- `3RD.POSCAR.XXXXX`: Displaced configurations

### Fourth-order  
- `FORCE_CONSTANTS_4TH`: Fourth-order force constants
- `4TH.SPOSCAR`: Supercell structure
- `4TH.POSCAR.XXXXX`: Displaced configurations

## Technical Details

The package uses:
- **Cython**: For performance-critical symmetry operations
- **spglib**: For crystal symmetry analysis
- **NumPy**: For array operations
- **Click**: For command-line interface

The symmetry reduction algorithm significantly reduces the number of required DFT calculations by identifying equivalent atomic displacements.

## Citation

If you use this software in your research, please cite the relevant publications for third-order and fourth-order force constant calculations.

## License

[Add license information here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and questions, please use the GitHub issue tracker.