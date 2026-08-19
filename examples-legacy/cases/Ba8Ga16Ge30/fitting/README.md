# Ba8Ga16Ge30 fitting data

This is the 54-atom `Ba8Ga16Ge30` clathrate training set from the public
hiPhive examples repository. It was retrieved with Git LFS, then converted
without atom reordering into MLFCS input files.

`input/training.extxyz` contains 200 DFT force snapshots. `input/primitive.vasp`
and `input/reference.vasp` are the same explicit 54-atom primitive calculation
frame, so the supercell matrix is the identity. `input/source.json` records the
upstream path and source database counts.

To reproduce the conversion after cloning the upstream data and running
`git lfs pull`:

```bash
uv run python convert_hiphive_data.py /path/to/hiphive-examples/examples/BaGaGe_clathrate
```

The upstream project is [materials-modeling/hiphive-examples](https://gitlab.com/materials-modeling/hiphive-examples).

## Matching Model 4

The upstream `construct_models` Model 4 uses only two-body clusters:

| Order | Cutoff | Maximum body order |
| --- | ---: | ---: |
| FC2 | 5.40 Å | 2 |
| FC3 | 4.35 Å | 2 |
| FC4 | 4.35 Å | 2 |

Run the corresponding unregularized MLFCS fit with:

```bash
uv run python examples/cases/Ba8Ga16Ge30/fitting/run.py
```

It keeps all 200 frames for training, as in the upstream least-squares model,
and writes native `mlfcs.h5`, phonopy `FORCE_CONSTANTS_2ND`, and ShengBTE
FC3/FC4 text files under `mlfcs/`.
