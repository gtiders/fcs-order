from mlfcs import write_force_constants
from pathlib import Path

from ase.io import read

from mlfcs.fitting import ForceConstantFitter

CASE = Path(__file__).resolve().parent


def main() -> None:
    fitter = ForceConstantFitter(
        read(CASE / "input/primitive.vasp"),
        read(CASE / "input/supercell.vasp"),
        orders=(2,),
        cutoffs={2: 5.4},
        max_body_orders={2: 2},
    )
    result = fitter.fit(
        read(CASE / "input/train.extxyz", index=":"),
        validation_split=0.0,
        acoustic_sum_rule=True,
        cache_directory=CASE / "results/cache",
    )
    output = CASE / "results"
    output.mkdir(parents=True, exist_ok=True)
    write_force_constants(result.force_constants, output / "mlfcs.h5", format="hdf5")
    write_force_constants(result.force_constants, output / "FORCE_CONSTANTS_2ND", format="phonopy", order=2)
    write_force_constants(result.force_constants, output / "fc2.h5", format="phonopy_hdf5", order=2)


if __name__ == "__main__":
    main()
