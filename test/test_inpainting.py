import os
import time
import proteopt
import prody
import numpy

from proteopt import rfdesign_inpainting
from proteopt.scaffold_problem import ScaffoldProblem

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


def test_basic():
    handle = prody.parsePDB(
        os.path.join(DATA_DIR, "1MBN.pdb"), model=1)

    start = time.time()
    runner = rfdesign_inpainting.RFDesignInpainting()
    print("*** Initialization time", time.time() - start)

    start = time.time()
    spec = (
        rfdesign_inpainting.ScaffoldProblem(handle)
        .add_segment(structure=handle.select("chain A and resid 1 to 21"))
        .add_variable_length_segment(10, 20)
        .add_segment(structure=handle.select("chain A and resid 34 to 150")))

    results = runner.run(
        spec,
        num=2,
        n_cycle=5)
    print("*** Run time %0.2f" % (time.time() - start))

    assert results.shape[0] == 2
    assert results.structure.map(lambda s: s.ca.getSequence()).nunique() == 2
    for _, row in results.iterrows():
        construct = row.structure
        seq = construct.ca.getSequence()
        print(seq)
        print(row.seq_A)

        assert len(seq) <= 159
        assert len(seq) >= 148

        assert seq[:21] == handle.select(
            "chain A and resid 1 to 21").ca.getSequence()
        assert seq[-117:] == handle.select(
            "chain A and resid 34 to 150").ca.getSequence()

    spec = (
        rfdesign_inpainting.ScaffoldProblem(handle)
        .add_segment(structure=handle.select("chain A and resid 1 to 21"))
        .add_segment(15)
        .add_segment(structure=handle.select("chain A and resid 34 to 150"))
        .add_segment(1)
    )

    results = runner.run(
        spec,
        num=2,
        n_cycle=5)
    print("*** Run time %0.2f" % (time.time() - start))

    assert results.shape[0] == 2
    assert results.structure.map(lambda s: s.ca.getSequence()).nunique() == 2
    for _, row in results.iterrows():
        construct = row.structure
        seq = construct.ca.getSequence()
        print(seq)
        print(row.seq_A)

        assert len(seq) == 154

        assert seq[:21] == handle.select(
            "chain A and resid 1 to 21").ca.getSequence()
        assert seq[-118:-1] == handle.select(
            "chain A and resid 34 to 150").ca.getSequence()
        assert seq[21:(21 + 15)] == construct.select(
            "unconstrained").ca.getSequence()[:-1]

