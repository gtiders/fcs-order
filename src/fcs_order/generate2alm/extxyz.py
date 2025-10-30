# fmt: off

"""
Extended XYZ support - Write-only version

Write files in "extended" XYZ format, storing additional
per-configuration information as key-value pairs on the XYZ
comment line, and additional per-atom properties as extra columns.

This version contains only write functionality, read functions have been removed.

Contributed by James Kermode <james.kermode@gmail.com>
"""
import json
import numbers
import re
import warnings
import numpy as np

from ase.calculators.calculator import all_properties
from ase.constraints import FixAtoms, FixCartesian
from ase.outputs import ArrayProperty, all_outputs
from ase.spacegroup.spacegroup import Spacegroup
from ase.stress import voigt_6_to_full_3x3_stress
from ase.utils import writer

__all__ = ['write_xyz', 'write_extxyz']

PROPERTY_NAME_MAP = {'positions': 'pos',
                     'numbers': 'Z',
                     'charges': 'charge',
                     'symbols': 'species'}

REV_PROPERTY_NAME_MAP = dict(zip(PROPERTY_NAME_MAP.values(),
                                 PROPERTY_NAME_MAP.keys()))

KEY_QUOTED_VALUE = re.compile(r'([A-Za-z_]+[A-Za-z0-9_-]*)'
                              + r'\s*=\s*["\{\}]([^"\{\}]+)["\{\}]\s*')
KEY_VALUE = re.compile(r'([A-Za-z_]+[A-Za-z0-9_]*)\s*='
                       + r'\s*([^\s]+)\s*')
KEY_RE = re.compile(r'([A-Za-z_]+[A-Za-z0-9_-]*)\s*')

UNPROCESSED_KEYS = {'uid'}

SPECIAL_3_3_KEYS = {'Lattice', 'virial', 'stress'}

# Determine 'per-atom' and 'per-config' based on all_outputs shape,
# but filter for things in all_properties because that's what
# SinglePointCalculator accepts
per_atom_properties = []
per_config_properties = []
for key, val in all_outputs.items():
    if key not in all_properties:
        continue
    if isinstance(val, ArrayProperty) and val.shapespec[0] == 'natoms':
        per_atom_properties.append(key)
    else:
        per_config_properties.append(key)


def escape(string):
    if (' ' in string or
            '"' in string or "'" in string or
            '{' in string or '}' in string or
            '[' in string or ']' in string):
        string = string.replace('"', '\\"')
        string = f'"{string}"'
    return string


def key_val_dict_to_str(dct, sep=' '):
    """
    Convert atoms.info dictionary to extended XYZ string representation
    """

    def array_to_string(key, val):
        # some ndarrays are special (special 3x3 keys, and scalars/vectors of
        # numbers or bools), handle them here
        if key in SPECIAL_3_3_KEYS:
            # special 3x3 matrix, flatten in Fortran order
            val = val.reshape(val.size, order='F')
        if val.dtype.kind in ['i', 'f', 'b']:
            # numerical or bool scalars/vectors are special, for backwards
            # compat.
            if len(val.shape) == 0:
                # scalar
                val = str(known_types_to_str(val))
            elif len(val.shape) == 1:
                # vector
                val = ' '.join(str(known_types_to_str(v)) for v in val)
        return val

    def known_types_to_str(val):
        if isinstance(val, (bool, np.bool_)):
            return 'T' if val else 'F'
        elif isinstance(val, numbers.Real):
            return f'{val}'
        elif isinstance(val, Spacegroup):
            return val.symbol
        else:
            return val

    if len(dct) == 0:
        return ''

    string = ''
    for key in dct:
        val = dct[key]

        if isinstance(val, np.ndarray):
            val = array_to_string(key, val)
        else:
            # convert any known types to string
            val = known_types_to_str(val)

        if val is not None and not isinstance(val, str):
            # what's left is an object, try using JSON
            if isinstance(val, np.ndarray):
                val = val.tolist()
            try:
                val = '_JSON ' + json.dumps(val)
                # if this fails, let give up
            except TypeError:
                warnings.warn('Skipping unhashable information '
                              '{}'.format(key))
                continue

        key = escape(key)  # escape and quote key
        eq = "="
        # Should this really be setting empty value that's going to be
        # interpreted as bool True?
        if val is None:
            val = ""
            eq = ""
        val = escape(val)  # escape and quote val

        string += f'{key}{eq}{val}{sep}'

    return string.strip()


def output_column_format(atoms, columns, arrays, write_info=True):
    """
    Helper function to build extended XYZ comment line
    """
    fmt_map = {'d': ('R', '%16.8f'),
               'f': ('R', '%16.8f'),
               'i': ('I', '%8d'),
               'O': ('S', '%s'),
               'S': ('S', '%s'),
               'U': ('S', '%-2s'),
               'b': ('L', ' %.1s')}

    # NB: Lattice is stored as tranpose of ASE cell,
    # with Fortran array ordering
    lattice_str = ('Lattice="'
                   + ' '.join([str(x) for x in np.reshape(atoms.cell.T,
                                                          9, order='F')]) +
                   '"')

    property_names = []
    property_types = []
    property_ncols = []
    dtypes = []
    formats = []

    for column in columns:
        array = arrays[column]
        dtype = array.dtype

        property_name = PROPERTY_NAME_MAP.get(column, column)
        property_type, fmt = fmt_map[dtype.kind]
        property_names.append(property_name)
        property_types.append(property_type)

        if (len(array.shape) == 1
                or (len(array.shape) == 2 and array.shape[1] == 1)):
            ncol = 1
            dtypes.append((column, dtype))
        else:
            ncol = array.shape[1]
            for c in range(ncol):
                dtypes.append((column + str(c), dtype))

        formats.extend([fmt] * ncol)
        property_ncols.append(ncol)

    props_str = ':'.join([':'.join(x) for x in
                          zip(property_names,
                              property_types,
                              [str(nc) for nc in property_ncols])])

    comment_str = ''
    if atoms.cell.any():
        comment_str += lattice_str + ' '
    comment_str += f'Properties={props_str}'

    info = {}
    if write_info:
        info.update(atoms.info)
    info['pbc'] = atoms.get_pbc()  # always save periodic boundary conditions
    comment_str += ' ' + key_val_dict_to_str(info)

    dtype = np.dtype(dtypes)
    fmt = ' '.join(formats) + '\n'

    return comment_str, property_ncols, dtype, fmt


def save_calc_results(atoms, calc=None, calc_prefix=None,
                      remove_atoms_calc=False, force=False):
    """Update information in atoms from results in a calculator

    Args:
    atoms (ase.atoms.Atoms): Atoms object, modified in place
    calc (ase.calculators.Calculator, optional): calculator to take results
        from.  Defaults to :attr:`atoms.calc`
    calc_prefix (str, optional): String to prefix to results names
        in :attr:`atoms.arrays` and :attr:`atoms.info`. Defaults to
        calculator class name
    remove_atoms_calc (bool): remove the calculator from the `atoms`
        object after saving its results.  Defaults to `False`, ignored if
        `calc` is passed in
    force (bool, optional): overwrite existing fields with same name,
        default False
    """
    if calc is None:
        calc_use = atoms.calc
    else:
        calc_use = calc

    if calc_use is None:
        return None, None

    if calc_prefix is None:
        calc_prefix = calc_use.__class__.__name__ + '_'

    per_config_results = {}
    per_atom_results = {}
    for prop, value in calc_use.results.items():
        if prop in per_config_properties:
            per_config_results[calc_prefix + prop] = value
        elif prop in per_atom_properties:
            per_atom_results[calc_prefix + prop] = value

    if not force:
        if any(key in atoms.info for key in per_config_results):
            raise KeyError("key from calculator already exists in atoms.info")
        if any(key in atoms.arrays for key in per_atom_results):
            raise KeyError("key from calculator already exists in atoms.arrays")

    atoms.info.update(per_config_results)
    atoms.arrays.update(per_atom_results)

    if remove_atoms_calc and calc is None:
        atoms.calc = None


@writer
def write_xyz(fileobj, images, comment='', columns=None,
              write_info=True,
              write_results=True, plain=False, vec_cell=False,
              if_no_atom_count=False):
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

    if hasattr(images, 'get_positions'):
        images = [images]

    for atoms in images:
        natoms = len(atoms)

        if write_results:
            calculator = atoms.calc
            atoms = atoms.copy()

            save_calc_results(atoms, calculator, calc_prefix="")

            if atoms.info.get('stress', np.array([])).shape == (6,):
                atoms.info['stress'] = \
                    voigt_6_to_full_3x3_stress(atoms.info['stress'])

        if columns is None:
            fr_cols = (['symbols', 'positions', 'move_mask']
                       + [key for key in atoms.arrays if
                          key not in ['symbols', 'positions', 'numbers',
                                      'species', 'pos']])
        else:
            fr_cols = columns[:]

        if vec_cell:
            plain = True

        if plain:
            fr_cols = ['symbols', 'positions']
            write_info = False
            write_results = False

        # Move symbols and positions to first two properties
        if 'symbols' in fr_cols:
            i = fr_cols.index('symbols')
            fr_cols[0], fr_cols[i] = fr_cols[i], fr_cols[0]

        if 'positions' in fr_cols:
            i = fr_cols.index('positions')
            fr_cols[1], fr_cols[i] = fr_cols[i], fr_cols[1]

        # Check first column "looks like" atomic symbols
        if fr_cols[0] in atoms.arrays:
            symbols = atoms.arrays[fr_cols[0]]
        else:
            symbols = [*atoms.symbols]

        if natoms > 0 and not isinstance(symbols[0], str):
            raise ValueError('First column must be symbols-like')

        # Check second column "looks like" atomic positions
        pos = atoms.arrays[fr_cols[1]]
        if pos.shape != (natoms, 3) or pos.dtype.kind != 'f':
            raise ValueError('Second column must be position-like')

        # if vec_cell add cell information as pseudo-atoms
        if vec_cell:
            nPBC = 0
            for i, b in enumerate(atoms.pbc):
                if not b:
                    continue
                nPBC += 1
                symbols.append('VEC' + str(nPBC))
                pos = np.vstack((pos, atoms.cell[i]))
            # add to natoms
            natoms += nPBC
            if pos.shape != (natoms, 3) or pos.dtype.kind != 'f':
                raise ValueError(
                    'Pseudo Atoms containing cell have bad coords')

        # Move mask
        if 'move_mask' in fr_cols:
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
                fr_cols.remove('move_mask')

        # Collect data to be written out
        arrays = {}
        for column in fr_cols:
            if column == 'positions':
                arrays[column] = pos
            elif column in atoms.arrays:
                arrays[column] = atoms.arrays[column]
            elif column == 'symbols':
                arrays[column] = np.array(symbols)
            elif column == 'move_mask':
                arrays[column] = cnstr
            else:
                raise ValueError(f'Missing array "{column}"')

        comm, ncols, dtype, fmt = output_column_format(atoms,
                                                       fr_cols,
                                                       arrays,
                                                       write_info)

        if plain or comment != '':
            # override key/value pairs with user-speficied comment string
            comm = comment.rstrip()
            if '\n' in comm:
                raise ValueError('Comment line should not have line breaks.')

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
        if not if_no_atom_count:
            fileobj.write('%d\n' % nat)
        for i in range(natoms):
            fileobj.write(f'# Snapshot: {i+1}, E_pot (Rydberg): {atoms.info["alm_potential_energy"]}\n')
            fileobj.write(fmt % tuple(data[i]))

write_extxyz = write_xyz
