"""Calculate several force-constant orders with a NEP89 model through calorine.

Example:
    uv run --with calorine python examples/nep89_orders.py POSCAR nep89.txt \
        --orders 2 3 4 --supercell 2 2 2 --cutoff -3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ase.io import read
from calorine.calculators import CPUNEP

from mlfcs import ForceConstantCalculation


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("structure", type=Path, help="ASE-readable primitive structure")
    parser.add_argument("model", type=Path, help="NEP89 model in nep.txt format")
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--supercell", type=int, nargs=3, default=(2, 2, 2))
    parser.add_argument("--cutoff", type=float, default=-3)
    parser.add_argument("--displacement", type=float, default=0.01)
    parser.add_argument("--jax-platform", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("force_constants"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    primitive = read(arguments.structure)
    calculator = CPUNEP(str(arguments.model))
    arguments.output.mkdir(parents=True, exist_ok=True)

    for order in arguments.orders:
        calculation = ForceConstantCalculation(
            primitive,
            order=order,
            supercell=tuple(arguments.supercell),
            cutoff=arguments.cutoff,
            displacement=arguments.displacement,
            jax_platform=arguments.jax_platform,
        )
        result = calculation.run(calculator, acoustic_sum_rule=True)
        result.write(
            arguments.output / f"force_constants_order_{order}.h5",
            format="hdf5",
            order=order,
        )


if __name__ == "__main__":
    main()
