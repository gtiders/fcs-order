import numpy as np

from ase.utils import writer
from ase.stress import voigt_6_to_full_3x3_stress
from ase.constraints import FixAtoms, FixCartesian

from ase.io.extxyz import save_calc_results, output_column_format


@writer
def write_xyz(
    fileobj,
    images,
    comment="",
    columns=None,
    write_info=True,
    write_results=True,
    plain=False,
    vec_cell=False,
    is_natoms=False,
    is_index_comment=False,
):
    """
    Write output in extended XYZ format

    Optionally, specify which columns (arrays) to include in output,
    whether to write the contents of the `atoms.info` dict to the
    XYZ comment line (default is True), the results of any
    calculator attached to this Atoms. The `plain` argument
    can be used to write a simple XYZ file with no additional information.
    `vec_cell` can be used to write the cell vectors as additional
    pseudo-atoms.

    See documentation for :func:`read_xyz()` for further details of the extended
    XYZ file format.
    """

    if hasattr(images, "get_positions"):
        images = [images]

    for index, atoms in enumerate(images):
        natoms = len(atoms)

        if write_results:
            calculator = atoms.calc
            atoms = atoms.copy()

            save_calc_results(atoms, calculator, calc_prefix="")

            if atoms.info.get("stress", np.array([])).shape == (6,):
                atoms.info["stress"] = voigt_6_to_full_3x3_stress(atoms.info["stress"])

        if columns is None:
            fr_cols = ["symbols", "positions", "move_mask"] + [
                key
                for key in atoms.arrays
                if key not in ["symbols", "positions", "numbers", "species", "pos"]
            ]
        else:
            fr_cols = columns[:]

        if vec_cell:
            plain = True

        if plain:
            fr_cols = ["symbols", "positions"]
            write_info = False
            write_results = False

        # Move symbols and positions to first two properties
        if "symbols" in fr_cols:
            i = fr_cols.index("symbols")
            fr_cols[0], fr_cols[i] = fr_cols[i], fr_cols[0]

        if "positions" in fr_cols:
            i = fr_cols.index("positions")
            fr_cols[1], fr_cols[i] = fr_cols[i], fr_cols[1]

        # Check first column "looks like" atomic symbols
        if fr_cols[0] in atoms.arrays:
            symbols = atoms.arrays[fr_cols[0]]
        else:
            symbols = [*atoms.symbols]

        if natoms > 0 and not isinstance(symbols[0], str):
            raise ValueError("First column must be symbols-like")

        # Check second column "looks like" atomic positions
        pos = atoms.arrays[fr_cols[1]]
        if pos.shape != (natoms, 3) or pos.dtype.kind != "f":
            raise ValueError("Second column must be position-like")

        # if vec_cell add cell information as pseudo-atoms
        if vec_cell:
            nPBC = 0
            for i, b in enumerate(atoms.pbc):
                if not b:
                    continue
                nPBC += 1
                symbols.append("VEC" + str(nPBC))
                pos = np.vstack((pos, atoms.cell[i]))
            # add to natoms
            natoms += nPBC
            if pos.shape != (natoms, 3) or pos.dtype.kind != "f":
                raise ValueError("Pseudo Atoms containing cell have bad coords")

        # Move mask
        if "move_mask" in fr_cols:
            cnstr = images[0].constraints
            if len(cnstr) > 0:
                c0 = cnstr[0]
                if isinstance(c0, FixAtoms):
                    cnstr = np.ones((natoms,), dtype=bool)
                    for idx in c0.index:
                        cnstr[idx] = False  # cnstr: atoms that can be moved
                elif isinstance(c0, FixCartesian):
                    masks = np.ones((natoms, 3), dtype=bool)
                    for i in range(len(cnstr)):
                        idx = cnstr[i].index
                        masks[idx] = cnstr[i].mask
                    cnstr = ~masks  # cnstr: coordinates that can be moved
            else:
                fr_cols.remove("move_mask")

        # Collect data to be written out
        arrays = {}
        for column in fr_cols:
            if column == "positions":
                arrays[column] = pos
            elif column in atoms.arrays:
                arrays[column] = atoms.arrays[column]
            elif column == "symbols":
                arrays[column] = np.array(symbols)
            elif column == "move_mask":
                arrays[column] = cnstr
            else:
                raise ValueError(f'Missing array "{column}"')

        comm, ncols, dtype, fmt = output_column_format(
            atoms, fr_cols, arrays, write_info
        )

        if plain or comment != "":
            # override key/value pairs with user-speficied comment string
            comm = comment.rstrip()
            if "\n" in comm:
                raise ValueError("Comment line should not have line breaks.")

        # Pack fr_cols into record array
        data = np.zeros(natoms, dtype)
        for column, ncol in zip(fr_cols, ncols):
            value = arrays[column]
            if ncol == 1:
                data[column] = np.squeeze(value)
            else:
                for c in range(ncol):
                    data[column + str(c)] = value[:, c]

        nat = natoms
        if vec_cell:
            nat -= nPBC
        # Write the output
        if is_natoms:
            fileobj.write("%d\n" % nat)
        if is_index_comment:
            fileobj.write(f"{comm}_snapshot{index}\n")

        for i in range(natoms):
            fileobj.write(fmt % tuple(data[i]))
