import os
import numpy
import pandas

import prody
import yabul

import proteopt
import proteopt.proteinmpnn

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

from .util import ALPHAFOLD_WEIGHTS_DIR


def test_basic():
    region1 = prody.parsePDB(
        os.path.join(DATA_DIR, "1MBN.pdb")
    ).select("protein chain A and resid 10 to 39")
    sequence = region1.ca.getSequence()
    fixed_region = region1.select("resid 25 to 28 or resid 35")

    runner = proteopt.proteinmpnn.ProteinMPNN()
    results = runner.run(region1, num=5, fixed=fixed_region)
    print(results)

    assert results.shape[0] == 5
    assert list(results.seq.str.len().unique()) == [len(region1.ca)]

    for _, row in results.iterrows():
        assert row.seq[:10] != sequence[:10]
        assert row.seq[15:19] == sequence[15:19]
        assert row.seq[25] == sequence[25]


def test_multiple_chains():
    full_handle = prody.parsePDB(os.path.join(DATA_DIR, "6JJP.pdb"))
    handle = full_handle.select(
        "protein and chain A B or (chain C and resid 75 to 150)")

    # fix the antibody sequence
    fixed = handle.select('chain A B or (chain C and resid 125 to 136)')

    runner = proteopt.proteinmpnn.ProteinMPNN()
    results = runner.run(handle, num=5, fixed=fixed)

    epitope_sequence = handle.select("chain A").ca.getSequence()
    for _, row in results.iterrows():
        print(row)
        assert row.seq_C[:10] != epitope_sequence[:10]
        assert row.seq_C[50:50 + 12] == full_handle.select("chain C and resid 125 to 136").ca.getSequence()

        assert row.seq_A.replace("X", "") == handle.select("chain A").ca.getSequence()
        assert row.seq_B.replace("X", "") == handle.select("chain B").ca.getSequence()

