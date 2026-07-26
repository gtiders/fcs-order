# mlfcs-new

An ASE-first API for symmetry-reduced finite-difference force constants. The runtime is
independent of phonopy, symfc, and any particular force calculator.

The calculation pipeline is order-parameterized. Third- and fourth-order calculations are the
tested production paths; higher orders use the same sparse cluster, finite-difference,
reconstruction, and HDF5 machinery but can become combinatorially expensive.

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
fc3 = calculation.reap(forces, acoustic_sum_rule=True)
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

To checkpoint calculator output before reconstruction:

```python
forces = calculation.evaluate(calculator)
np.savez_compressed("forces.npz", forces=forces, plan_hash=calculation.plan.hash)
fc3 = calculation.reap(forces, plan_hash=calculation.plan.hash)
```

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

- `hdf5`: sparse cluster tensors, structure, metadata, and ordering arrays; any order.
- `numpy` or `npz`: materialized compact NumPy tensors, subject to a memory budget.
- `shengbte`: third- and fourth-order text output.

The ShengBTE writer emits, for order 3 or 4:

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

Reconstruction remains sparse until dense values are requested:

```python
fc5.write("fc5.h5", format="hdf5")       # no dense N**4 allocation
dense = fc5.materialize(5)                # checks the default 2 GB budget
dense = fc5.materialize(5, max_bytes=None)  # explicit opt-out
```

## Acoustic sum rule

Translational invariance is imposed independently for each force-constant order. For order `n`,
the first `n - 1` atom indices and all Cartesian components are fixed while the final atom index
is summed. Permutation symmetry provides the equivalent constraints on the other atom axes.

```python
constrained = calculation.reap(forces, acoustic_sum_rule=True)
raw = calculation.reap(forces, acoustic_sum_rule=False)
```

The constrained path constructs a sparse matrix `A` in the independent orbit-parameter space
and orthogonally projects the measured parameters onto `null(A)` with a strict iterative solve.
The third- and fourth-order tests require a final dense ASR residual below `1e-10`.

## Numerical reference status

Writer ordering, orbit planning, raw force reconstruction, and ASR projection are tested
separately.

When the same captured IFC values were passed to both implementations, the previous third-order
text format and the new order-parameterized writer matched byte-for-byte. Fourth-order block,
translation, atom, and Cartesian-component order also matched; fourth-order numeric formatting
is now intentionally scientific notation at the user's request.

The force-constant values are intentionally not compatible with the previous ASR projection.
ALAMODE and hiphive both define order-`n` translational invariance as a sum over one atom axis.
The previous fourth-order implementation sums two atom axes together, while its third-order
implementation sums one. The previous relative-weight projection can also amplify raw IFCs.

For saved Si 2x2x2 fifth-shell NEP forces, the strict implementation gives:

- FC3: 11 orbits, 72 configurations, maximum ASR residual `3.89e-15`.
- FC4: 41 orbits, 1056 configurations, maximum ASR residual `2.54e-14`.
- Strict FC4 versus previous FC4: maximum difference `1.31e-3`, RMS `5.71e-5`.

Strict FC3 differs substantially from the previous projected FC3 because the previous projection
amplifies the raw maximum IFC from about `8.71` to `34.30`; the strict solution remains about
`8.37`. This comparison is an ASR-method difference, not an atom- or file-order mismatch.

Complex multi-species fourth-order planning is also checked at a three-neighbor cutoff. The new
and reference implementations agree on the cutoff, irreducible-cluster count, and force-job
count:

- K3Au3Sb2: 61 clusters and 3568 configurations.
- KAsPt: 45 clusters and 2936 configurations.
- NaS: 43 clusters and 2016 configurations.

The full NaS NEP run was also used to isolate the previous fourth-order double-axis ASR behavior.
It is retained as a black-box research fixture rather than a compatibility target.

An order-5 end-to-end smoke calculation was subsequently completed for NaS 2x2x2 with a
first-neighbor cutoff. It contains 16 irreducible cluster orbits, 403 independent parameters,
2432 force configurations, and 1686 reconstructed sparse cluster images. The HDF5 file is about
789 KiB; its equivalent dense tensor would require 243 GiB. Peak resident memory during sparse
reconstruction was about 1.06 GiB. This validates the generic high-order path but does not replace
the production third- and fourth-order regression suite.

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
