from collections import namedtuple

import numpy
import pandas
import prody

AlignmentResult = namedtuple("AlignmentResult", ["aligned", "rmsd"])

# equivalent atoms (symmetries)
# Taken from alphafold
EQUIVALENT_ATOMS = {
    "ASP": [("OD1", "OD2")],
    "GLU": [("OE1", "OE2")],
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
}

def alternative_atom_name(resname, atom_name):
    """
    Get the alternative atom name for a given atom name, if it exists.
    """
    pairs = EQUIVALENT_ATOMS.get(resname)
    if not pairs:
        return None
    matching = [p for p in pairs if atom_name in p]
    if not matching:
        return None
    pair, = matching
    return pair[0] if pair[0] != atom_name else pair[1]



def smart_align(
        mobile : prody.AtomGroup,
        target : prody.AtomGroup,
        part_of_mobile_to_align  : prody.AtomGroup = None,
        reorder_atoms_by_name : bool = True,
        fix_symmetries : bool = True):
    """
    Align mobile atoms onto target atoms, optionally taking into account
    atom names and symmetries.

    Parameters
    ----------
    mobile : prody.AtomGroup
        The mobile atoms to align.
    target : prody.AtomGroup
        The target atoms to align to.
    part_of_mobile_to_align : prody.AtomGroup
        The part of the mobile atoms to align. If None, all atoms in the
        mobile are aligned.
    reorder_atoms_by_name : bool
        If True, reorder the atoms in part_of_mobile_to_align to match the
        order of the target atoms, based on atom names. Note: this does NOT
        handle differences in residue order.
    fix_symmetries : bool
        If True, fix symmetries by swapping equivalent atoms in the
        part_of_mobile_to_align (e.g. ASP OD1 and OD2) if it improves rmsd.

    Returns
    -------
    (aligned mobile atoms, rmsd)
    """

    if part_of_mobile_to_align is None:
        part_of_mobile_to_align = mobile.copy()

    if len(part_of_mobile_to_align) != len(target):
        raise ValueError(
            "Differing number of atoms between mobile and target: %d != %d" % (
                len(part_of_mobile_to_align), len(target)))

    if reorder_atoms_by_name:
        remapped_resindices_mobile, _ = pandas.Series(
            part_of_mobile_to_align.getResindices()).factorize()
        remapped_resindices_target, _ = pandas.Series(
            part_of_mobile_to_align.getResindices()).factorize()

        mobile_keys = list(zip(
            remapped_resindices_mobile, part_of_mobile_to_align.getNames()))
        target_keys = list(zip(
            remapped_resindices_target, target.getNames()))

        missing_mobile = [
            pair for pair in target_keys if pair not in set(mobile_keys)
        ]
        missing_target = [
            pair for pair in mobile_keys if pair not in set(target_keys)
        ]
        if missing_mobile:
            raise ValueError(
                "Missing atoms in mobile: %s. Mobile atoms are: %s" % (
                    str(missing_mobile),
                    str(mobile_keys)))
        if missing_target:
            raise ValueError(
                    "Missing atoms in target: %s. Target atoms are: %s" % (
                        str(missing_target),
                        str(target_keys)))

        key_to_mobile_index = dict(
            (key, i) for (i, key) in enumerate(mobile_keys))
        indices_into_part_of_mobile = [
            key_to_mobile_index[key] for key in target_keys
        ]
    else:
        indices_into_part_of_mobile = numpy.arange(len(part_of_mobile_to_align))

    if fix_symmetries:
        df = pandas.DataFrame({
            "resindex": part_of_mobile_to_align.getResindices(),
            "resname": part_of_mobile_to_align.getResnames(),
            "atom_name": part_of_mobile_to_align.getNames(),
        })
        df["alt_name"] = [
            alternative_atom_name(row.resname, row.atom_name)
            for _, row in df.iterrows()
        ]
        df.index.name = "original_index"
        df = df.iloc[indices_into_part_of_mobile].reset_index()
        df = df.reset_index()

        df["swap_group"] = [
            "%d-%s-%s" % (
                row.resindex,
                row.resname,
                sorted([row.atom_name, row.alt_name])[0])
            if row.alt_name is not None else None
            for _, row in df.iterrows()
        ]
        swappable_df = df.loc[~df.alt_name.isnull()]
        possible_swaps = swappable_df.groupby("swap_group").index.unique().tolist()
    else:
        possible_swaps = []

    target_coords = target.getCoords()
    indices_into_part_of_mobile = numpy.array(indices_into_part_of_mobile)

    # We pick swaps greedily, which should be accurate enough
    best_rmsd = None
    best_transformation = None
    for swap_pair in [None] + possible_swaps:
        # Make the swap:
        if swap_pair is not None:
            (idx1, idx2) = swap_pair
            indices_into_part_of_mobile[[idx1, idx2]] = (
                indices_into_part_of_mobile[[idx2, idx1]])

        part_of_mobile_to_align_coords = numpy.take(
            part_of_mobile_to_align.getCoords(),
            indices_into_part_of_mobile,
            axis=0)

        transformation = prody.calcTransformation(
            part_of_mobile_to_align_coords, target_coords)
        transformed_coords = transformation.apply(part_of_mobile_to_align_coords)
        rmsd = prody.calcRMSD(transformed_coords, target_coords)

        if best_rmsd is None or rmsd < best_rmsd:
            best_rmsd = rmsd
            best_transformation = transformation
        elif swap_pair is not None:
            # Undo the swap if it didn't improve rmsd
            indices_into_part_of_mobile[[idx1, idx2]] = (
                indices_into_part_of_mobile[[idx2, idx1]])

    new_mobile = mobile.copy()
    best_transformation.apply(new_mobile)
    return AlignmentResult(new_mobile, best_rmsd)