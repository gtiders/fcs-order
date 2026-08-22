# Graphene: FC2 physical constraints

`input/phonopy_snapshot.extxyz` is one combined phonopy displacement
snapshot, not a multi-snapshot fitting dataset. `2_mlfcs_fit.py` nevertheless
provides a deliberately limited single-snapshot FC2 comparison using the same
8 Angstrom cutoff as the hiphive example.

`mlfcs/asr` contains the ASR fit. `mlfcs/born-huang-huang` applies strict FC2
Born-Huang and Huang postprocessing to the same result. The new projection
uses all tied nearest images and records its residuals in `metrics.json`.

Run the MLFCS comparison and combined plot with:

```bash
uv run python run.py
uv run --with phonopy --with seekpath --with matplotlib python ../../../plot_phonon_band.py \
  --supercell input/reference.vasp --force-constants mlfcs/born-huang-huang/FORCE_CONSTANTS_2ND \
  --output mlfcs/born-huang-huang/phonon-band.png
```
