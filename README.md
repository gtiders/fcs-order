# FCS-Order

Repository: [https://github.com/gtiders/fcs-order](https://github.com/gtiders/fcs-order)

A comprehensive Python package for calculating third-order and fourth-order force constants using finite displacement methods, with support for machine learning potentials and thermal disorder generation.

## Features

- **Third-order force constants**: Calculate 3-phonon interactions
- **Fourth-order force constants**: Calculate 4-phonon interactions  
- **Machine Learning Integration**: Direct calculation using ML potentials (NEP, DeepMD, HiPhive, Polymlp)
- **VASP Compatibility**: Full integration with VASP DFT calculations
- **Thermal Disorder Generation**: Create phonon-rattled structures at finite temperatures

## Installation

```bash
pip install fcs-order
```

Or install from source:

```bash
git clone https://github.com/your-repo/fcs-order.git
cd fcs-order
pip install -e .
```

## Available Commands

### Core Force Constant Commands

#### 1. Third-order Force Constants (`sow3` & `reap3`)

**Generate displaced structures:**
```bash
fcsorder sow3 <na> <nb> <nc> --cutoff <cutoff>
```

**Extract force constants from VASP results:**
```bash
fcsorder reap3 <na> <nb> <nc> --cutoff <cutoff> vasprun.0001.xml vasprun.0002.xml ...
```

Parameters:
- `na, nb, nc`: Supercell dimensions (expansion factors in a, b, c directions)
- `--cutoff`: Interaction cutoff (negative for nearest neighbors like -8, positive for distance in nm like 0.5)
- `vasprun.xml files`: VASP calculation results in order

#### 2. Fourth-order Force Constants (`sow4` & `reap4`)

**Generate displaced structures:**
```bash
fcsorder sow4 <na> <nb> <nc> --cutoff <cutoff>
```

**Extract force constants from VASP results:**
```bash
fcsorder reap4 <na> <nb> <nc> --cutoff <cutoff> vasprun.0001.xml vasprun.0002.xml ...
```

Parameters: Same as third-order commands

### Machine Learning Potential Commands

#### 3. ML Third-order Force Constants (`mlp3`)

```bash
fcsorder mlp3 <na> <nb> <nc> --cutoff <cutoff> --calc <calculator> --potential <potential_file>
```

#### 4. ML Fourth-order Force Constants (`mlp4`)

```bash
fcsorder mlp4 <na> <nb> <nc> --cutoff <cutoff> --calc <calculator> --potential <potential_file>
```

Parameters:
- `--calc`: Calculator type (`nep`, `dp`, `hiphive`, `polymlp`)
- `--potential`: Path to potential file (format depends on calculator)
- `--if_write`: Optional flag to save intermediate files

Supported ML Potentials:
- **NEP**: NEP potential (file: `nep.txt`)
- **DeepMD**: Deep Potential (file: `model.pb`)  
- **HiPhive**: HiPhive potential (file: `potential.fcp`)
- **Polymlp**: Polynomial ML potential (file: `polymlp.yaml`)

### Phonon Rattling Command

#### 5. Generate Thermally Disordered Structures (`phonon-rattle`)

```bash
fcsorder phonon-rattle <SPOSCAR> <fc2_file> [options]
```

Parameters:
- `SPOSCAR`: Supercell structure file
- `fc2_file`: Second-order force constants file (2nd, fc2, or FORCE_CONSTANTS_2ND)

Options:
- `--temperature, -t`: Temperature in Kelvin (default: 300.0)
- `--n_structures, -n`: Number of structures to generate (default: 100)
- `--max_disp`: Maximum displacement in Ångströms (default: 0.5)
- `--min_distance`: Minimum atomic distance in Ångströms (default: 1.5)
- `--batch_size`: Batch size for generation (default: 5000)
- `--if_qm`: Enable quantum statistics (default: True)
- `--imag_freq_factor`: Imaginary frequency scaling factor (default: 1.0)

Output: Saves valid structures to `structures_phonon_rattle_T<temperature>.xyz`

## Usage Examples

### Basic Third-order Calculation Workflow

1. Generate displaced structures:
```bash
fcsorder sow3 2 2 2 --cutoff -8
```

2. Run VASP calculations on generated 3RD.POSCAR.* files

3. Extract force constants:
```bash
fcsorder reap3 2 2 2 --cutoff -8 vasprun.*.xml
```

### Machine Learning Potential Calculation

```bash
fcsorder mlp3 4 4 4 --cutoff 0.8 --calc nep --potential nep.txt
```

### Phonon Rattling at High Temperature

```bash
fcsorder phonon-rattle SPOSCAR FORCE_CONSTANTS_2ND --temperature 800 --n_structures 200 --max_disp 0.8
```

## File Formats

- **SPOSCAR**: VASP structure format for supercells
- **FORCE_CONSTANTS_3RD**: Third-order force constants output
- **FORCE_CONSTANTS_4TH**: Fourth-order force constants output  
- **3RD.POSCAR.***: Displaced structures for 3-phonon calculations
- **4TH.POSCAR.***: Displaced structures for 4-phonon calculations
- **.xyz files**: Extended XYZ format for rattled structures

## Requirements

- Python 3.9+
- NumPy
- Click
- spglib
- VASP (for DFT calculations)
- Machine learning potential packages (optional)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use FCS-Order in your research, please cite:


## Support

For issues and questions, please open an issue on GitHub or contact the development team.
