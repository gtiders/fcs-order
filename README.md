# mlfcs-new

An ASE-first API for symmetry-reduced finite-difference force constants. The runtime is
independent of phonopy, symfc, and any particular force calculator.

Third- and fourth-order reconstruction is currently supported. The mixed central-difference
stencil and ShengBTE writer are order-parameterized.

## Units

- Structures and displacements: angstrom.
- Forces: eV/angstrom.
- Order-`n` force constants: `eV/angstrom^n`.
- Positive cutoff: radius in angstrom.
- Negative integer cutoff, for example `-5`: neighbor shell.
- JAX numerical kernels use 64-bit floating point.

## Sow and reap

```python
import numpy as np
from ase.io import read
from mlfcs import ForceConstantCalculation

calculation = ForceConstantCalculation(
    read("structures/POSCAR-Si.vasp"),
    order=3,
    supercell=(2, 2, 2),
    cutoff=-5,
    displacement=0.01,
)

structures = calculation.sow()
```

`sow()` returns `list[ase.Atoms]`. Its list order is the positional `reap()` contract:

```python
forces = np.asarray(load_forces_in_the_same_order())
fc3 = calculation.reap(forces)
```

The required shape is:

```text
(len(calculation.sow()), len(calculation.supercell), 3)
```

Every sow structure contains:

```python
atoms.info["mlfcs_configuration_id"]  # zero-based list position
atoms.info["mlfcs_plan_hash"]
atoms.info["mlfcs_atom_order"]
atoms.arrays["mlfcs_displacement"]
```

When force jobs return out of order, pass an ID mapping. Mapping insertion order is irrelevant:

```python
forces_by_id = {
    configuration_id: force_array,
    # ...
}

fc3 = calculation.reap(
    forces_by_id,
    plan_hash=calculation.plan.hash,
)
```

Missing IDs, extra IDs, invalid shapes, NaN/Inf, and mismatched plan hashes are rejected.
MLFCS does not parse or assume how forces were calculated.

## Optional direct ASE Calculator use

Users may supply any ASE Calculator directly. MLFCS adds no dependency on calculator packages:

```python
calculator = make_my_ase_calculator()
fc3 = calculation.run(calculator)
```

This evaluates the same sow list serially and passes the resulting force array to `reap()`.
External scheduling, checkpointing, and parallel execution remain user responsibilities.

## Atom ordering

The internal supercell order is:

```text
z → y → x → primitive_atom
```

The primitive-atom index is fastest. This is also the default order returned by `sow()`.

For primitive-atom-grouped structures:

```python
structures = calculation.sow(atom_order="grouped")
fc3 = calculation.reap(grouped_forces, atom_order="grouped")
```

Explicit mappings are available:

```python
calculation.index.grouped_permutation
calculation.index.internal_from_grouped
calculation.index.group_atoms(atoms)
```

## Force-constant I/O

Output format is always explicit:

```python
fc3.write("fc3.h5", format="hdf5")
fc3.write("fc3.npz", format="numpy")
fc3.write("FORCE_CONSTANTS_3RD", format="shengbte")
```

Available formats:

- `hdf5`: compact tensors, structure, metadata, and ordering arrays.
- `numpy` or `npz`: compact NumPy archive.
- `shengbte`: order-parameterized text output.

The ShengBTE writer emits, for order `n`:

- `n - 1` lattice-translation vectors per block;
- `n` primitive atom indices;
- `3**n` Cartesian components;
- scientific notation for every order.

HDF5 stores the following under `ordering/`:

```text
primitive_index
cell_translation
primitive_scaled_position
```

## Numerical reference status

Writer ordering and force-constant calculation are tested separately.

When the same captured IFC values were passed to both implementations, the previous third-order
text format and the new order-parameterized writer matched byte-for-byte. Fourth-order block,
translation, atom, and Cartesian-component order also matched; fourth-order numeric formatting
is now intentionally scientific notation at the user's request.

The force-constant calculations themselves are not byte-identical. For the Si 2x2x2, fifth-shell
NEP reference used during development:

- FC3 RMS difference: `9.50e-6 eV/angstrom^3`.
- FC4 RMS difference: `5.76e-5 eV/angstrom^4`.
- FC4 maximum absolute difference: `5.14e-3 eV/angstrom^4`.

The FC4 difference comes from three periodic-image representative choices in the previous
fourth-order traversal. Both calculations contain 41 cluster orbits and 750 independent tensor
parameters.

An additional end-to-end file comparison uses independent pipelines: the previous calculation
is written by its own writer, while the new calculation is written by the generic ShengBTE
writer. Block numbers, translations, atom indices, and Cartesian-component order match exactly.
Comparing every numeric component in the resulting files gives:

- FC3: 512 blocks, maximum absolute difference `5.169151e-5`, RMS `8.735334e-6`.
- FC4: 8072 blocks, maximum absolute difference `5.141029e-3`, RMS `5.653504e-5`.

These file-level RMS values include every written component, so they differ slightly from the
sparse reference-component metrics above.

Complex multi-species fourth-order planning is also checked at a three-neighbor cutoff. The new
and reference implementations agree on the cutoff, irreducible-cluster count, and force-job
count:

- K3Au3Sb2: 61 clusters and 3568 configurations.
- KAsPt: 45 clusters and 2936 configurations.
- NaS: 43 clusters and 2016 configurations.

The NaS end-to-end NEP calculation was run through both independent pipelines. Its FC4 values
are not numerically compatible: the maximum absolute difference is `11.90765 eV/angstrom^4`
and the RMS difference is `0.182445 eV/angstrom^4`. Inspection shows that the previous FC4 ASR
matrix sums over two atom axes together, whereas its FC3 matrix and the new order-parameterized
implementation sum one atom axis. For the standard last-atom ASR, the new NaS FC4 residual has
maximum/RMS `0.04261/0.000469`; the previous result has `8.03893/0.04302`.

## Development

All commands use uv and tests run serially:

```bash
uv sync
uv run pytest
uv run ruff check src tests tools
uv run ruff format --check src tests tools
```

The black-box reference helpers under `tools/` use a separate environment. They are not runtime
dependencies of the package.
