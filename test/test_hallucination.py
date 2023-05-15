import os
import time
import functools

import proteopt
import prody
import numpy

from rfdesign.hallucination import loss
import torch


from proteopt import rfdesign_hallucination
from proteopt.scaffold_problem import ScaffoldProblem

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


def test_basic():
    handle = prody.parsePDB(os.path.join(DATA_DIR, "1MBN.pdb"), model=1)
    print(len(handle))

    structure_to_recapitulate = handle.select("chain A and resid 22 to 33")
    print(len(structure_to_recapitulate.ca), structure_to_recapitulate.ca.getSequence())

    problem = ScaffoldProblem(handle)
    problem.add_segment(length=16)
    problem.add_segment(sequence="DPSK", length=4)
    problem.add_segment(
        structure=structure_to_recapitulate,
        sequence=structure_to_recapitulate.ca.getSequence(),
    )
    problem.add_segment(sequence="VTLADAGF")
    problem.add_segment(length=22)

    start = time.time()
    runner = rfdesign_hallucination.RFDesignHallucination()
    print("*** Initialization time", time.time() - start)

    for i in range(2):
        start = time.time()
        results = runner.run(
            problem,
            num=1,
            steps="g10,m10",
            w_crmsd=1,
            w_rog=1)
        print("*** Run time #%d" % (i + 1), time.time() - start)

        assert len(results) == 1

        construct = results.iloc[0].structure
        seq = construct.ca.getSequence()

        assert len(seq) == 62
        assert seq[20:20 + 12] == handle.select(
            "chain A and resid 22 to 33").ca.getSequence()
        assert seq[16:20] == "DPSK"
        assert seq[32:40] == "VTLADAGF"

        assert construct.select("constrained_by_sequence").ca.getSequence() == (
                "DPSK" + structure_to_recapitulate.ca.getSequence() + "VTLADAGF")
        assert construct.select("constrained_by_structure").ca.getSequence() == (
                structure_to_recapitulate.ca.getSequence())
        assert len(construct.select("unconstrained").ca) == 16 + 22
