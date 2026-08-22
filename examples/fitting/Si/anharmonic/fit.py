from mlfcs import write_force_constants
from pathlib import Path
from time import perf_counter

from ase.io import read

from mlfcs.fitting import ForceConstantFitter

CASE = Path(__file__).resolve().parent


def timed_write(force_constants, target: Path, **kwargs) -> None:
    started = perf_counter()
    print(f"TIMING io_start target={target.name} format={kwargs['format']}", flush=True)
    write_force_constants(force_constants, target, **kwargs)
    print(
        f"TIMING io_done target={target.name} format={kwargs['format']} "
        f"seconds={perf_counter() - started:.6f} bytes={target.stat().st_size}",
        flush=True,
    )


def main() -> None:
    fitter = ForceConstantFitter(
        read(CASE / "input/primitive.vasp"),
        read(CASE / "input/supercell.vasp"),
        orders=(2, 3, 4),
        # FC4 is restricted to the third Si neighbour shell (4.503 Å);
        # 4.6 Å lies safely between the third and fourth shells.
        cutoffs={2: 5.4, 3: 5.4, 4: 4.6},
        max_body_orders={2: 2, 3: 3, 4: 3},
    )
    result = fitter.fit(
        read(CASE / "input/train.extxyz", index=":"),
        validation_split=0.0,
        batch_size=4,
        regularization="scaled_group_lasso",
        acoustic_sum_rule=True,
        cache_directory=CASE / "results/cache",
        max_iterations=10_000,
    )
    output = CASE / "results"
    output.mkdir(parents=True, exist_ok=True)
    timed_write(result.force_constants, output / "mlfcs.h5", format="hdf5")
    timed_write(result.force_constants, output / "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
    timed_write(result.force_constants, output / "fc2.h5", format="phonopy_hdf5", order=2)
    timed_write(result.force_constants, output / "FORCE_CONSTANTS_3RD", format="shengbte", order=3)
    timed_write(result.force_constants, output / "fc3.h5", format="phono3py_hdf5", order=3)
    timed_write(result.force_constants, output / "FORCE_CONSTANTS_4TH", format="shengbte", order=4)


if __name__ == "__main__":
    main()
