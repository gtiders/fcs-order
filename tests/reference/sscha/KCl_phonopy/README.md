# KCl SSCHA comparison

English | [中文](README_ZH.md)

This directory is a self-contained end-to-end comparison between native MLFCS SSCHA and the
official phonopy KCl SSCHA reference.

- [`COMPARISON.md`](COMPARISON.md) records the numerical conditions, results, interpretation, and
  limitations.
- [`data/`](data/) contains the pinned upstream potential, structures, published reference FC2,
  license, and provenance notes.
- `test_kcl_potential.py` runs the potential through MLFCS and checks the physical scale.
- `test_provenance.py` verifies every imported artifact by SHA-256.
- `case.py` defines the exact conventional KCl cell used by the comparison.

Run this reference independently to keep its pypolymlp and JAX memory isolated:

```bash
uv run pytest tests/reference/sscha/KCl_phonopy
```
