# Repository Guidelines

## Project Structure & Module Organization
This repository uses a `src/` layout:

- `src/mlfcs/`: main Python package.
- `src/mlfcs/thirdorder/` and `src/mlfcs/fourthorder/`: CLI workflows and Cython/C++ acceleration for anharmonic force constants.
- `src/mlfcs/phonon.py`, `src/mlfcs/sscha.py`, `src/mlfcs/symmetry.py`: harmonic phonon, SSCHA, and symmetry utilities.
- `README.md` and `README_ZH.md`: user-facing documentation.
- `pyproject.toml` and `setup.py`: packaging/build configuration.

Prefer editing `.py`/`.pyx` sources; treat generated `.cpp` extension outputs as build artifacts unless a change explicitly requires regenerating them.

## Build, Test, and Development Commands
- `pip install .`: build extensions and install package.
- `pip install -e .`: editable install for development.
- `uv sync --dev`: create/update local dev environment from `pyproject.toml` and `uv.lock`.
- `python -m build`: build sdist/wheel for release checks.
- `thirdorder --help` and `fourthorder --help`: quick CLI sanity checks after changes.

Example local loop:
```bash
uv sync --dev
pip install -e .
thirdorder --help
```

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation.
- Use `snake_case` for functions/variables, `PascalCase` for classes (e.g., `ThirdOrderRun`).
- Keep CLI option names consistent with existing commands (`--forces`, `--hstep`, `--symprec`).
- Prefer small, focused changes and preserve existing public API names.

## Testing Guidelines
There is currently no dedicated `tests/` suite or configured pytest target. For contributions, include at least:

- CLI smoke validation (`thirdorder --help`, `fourthorder --help`).
- A minimal functional run on a known structure when touching core logic.
- Clear reproduction steps in the PR description.

If you add automated tests, place them in a new `tests/` directory and use `test_*.py` naming.

## Commit & Pull Request Guidelines
Recent history follows mostly Conventional Commit prefixes (`feat:`, `docs:`, `build:`). Use:

- `type(scope): short imperative summary`
- Keep subject lines concise and specific.

For pull requests, include:

- What changed and why.
- Any CLI/API impact.
- Validation commands run and key output.
- Linked issue(s) when applicable.
