# MoS2 monolayer: FC2 physical constraints

`input/training.extxyz` contains five ordinary ASE snapshots with explicit displaced
positions and forces. The reference is an `8 x 8 x 1` supercell of the explicit
three-atom `input/primitive.vasp`, with the periodic origin preserved from the source
data. `input/reference.vasp` fixes the atom labels used by every snapshot and output.

Run both fits from the repository root:

```bash
uv run python examples/cases/rotational_sum_rules/MoS2_monolayer/run.py
```

`mlfcs/asr` contains the ASR-constrained fit. `mlfcs/born-huang-huang` applies
strict FC2 Born-Huang and Huang postprocessing to that same fitted result.
Each output directory contains native sparse `mlfcs.h5`, phonopy text
`FORCE_CONSTANTS_2ND`, and `metrics.json`. The latter records both fit metrics
and physical constraint residuals. FC3 and higher orders are not involved.

Generate the two phonon-band views with:

```bash
uv run --with phonopy --with seekpath --with matplotlib python examples/plot_phonon_band.py \
  --supercell examples/cases/rotational_sum_rules/MoS2_monolayer/input/reference.vasp \
  --force-constants examples/cases/rotational_sum_rules/MoS2_monolayer/mlfcs/asr/FORCE_CONSTANTS_2ND \
  --output examples/cases/rotational_sum_rules/MoS2_monolayer/mlfcs/asr/phonon-band.png
```

Replace `asr` with `born-huang-huang` for the strict projection.
