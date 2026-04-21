# MLFCS Quickstart

This page gives the shortest runnable paths for CLI and Python APIs.

## 1) Install

```bash
git clone https://github.com/gtiders/mlfcs.git
cd mlfcs
pip install .
```

Requires Python `>=3.12`.

## 2) CLI: thirdorder/fourthorder

### Discover interfaces

```bash
thirdorder interfaces
fourthorder interfaces
```

### thirdorder minimal flow

```bash
# generate displacements
thirdorder sow 4 4 4 --cutoff -3 --format vasp

# run your calculator externally for each displaced structure...

# reap IFCs
thirdorder reap 4 4 4 --cutoff -3 \
  --forces $(find ./3RD_runs -name "vasprun.xml" | sort -V) \
  --forces-interface vasp
```

Output: `FORCE_CONSTANTS_3RD`.

### fourthorder minimal flow

```bash
fourthorder sow 3 3 3 --cutoff -2 --format vasp
fourthorder reap 3 3 3 --cutoff -2 \
  --forces $(find ./4TH_runs -name "vasprun.xml" | sort -V) \
  --forces-interface vasp
```

Output: `FORCE_CONSTANTS_4TH`.

## 3) Python API: secondorder

```python
from ase.io import read
from calorine.calculators import CPUNEP
from mlfcs.secondorder import MLPHONON, MLPSSCHA

calc = CPUNEP("nep.txt")
prim = read("POSCAR")

# harmonic phonon
phonon = MLPHONON(
    structure=prim,
    calculator=calc,
    supercell_matrix=[2, 2, 2],
    kwargs_generate_displacements={"distance": 0.01},
)
phonon.run()
phonon.write("FORCE_CONSTANTS")   # text
phonon.write("fc2.hdf5")          # hdf5 (auto by suffix)

# sscha
sscha = MLPSSCHA(
    unitcell=prim,
    calculator=calc,
    supercell_matrix=[3, 3, 3],
    temperature=300,
    number_of_snapshots=1000,
    max_iterations=20,
    fc_output="fc2_sscha.hdf5",
    fc_output_format="hdf5",
)
sscha.run()
```

## 4) Python API: hifinit

```python
from ase.io import read
from calorine.calculators import CPUNEP
from mlfcs.hifinit import HifinitRun

calc = CPUNEP("nep.txt")
prim = read("POSCAR")
supercell = read("SPOSCAR")

runner = HifinitRun(
    primitive=prim,
    supercell=supercell,
    calculator=calc,
    displacement=0.005,
    cutoffs=[None, None, 4.0],
)
runner.run(out_dir="./hifinit_results", verbose=True)
```

Typical outputs:
- `FORCE_CONSTANTS_2ND`, `FORCE_CONSTANTS_3RD`, `FORCE_CONSTANTS_4TH`
- `fc2.hdf5`, `fc3.hdf5`
