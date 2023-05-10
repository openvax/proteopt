import os
import time
import functools

import proteopt
import prody
import numpy

import torch

from proteopt import rfdiffusion
from proteopt.scaffold_problem import ScaffoldProblem

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


def test_basic():
    handle = prody.parsePDB(os.path.join(DATA_DIR, "1MBN.pdb"), model=1)
    print(len(handle))

    structure_to_recapitulate = handle.select("chain A and resid 22 to 33")
    print(len(structure_to_recapitulate.ca), structure_to_recapitulate.ca.getSequence())

    problem = ScaffoldProblem(handle)
    problem.add_fixed_length_segment(length=20)
    problem.add_fixed_length_segment(
        structure=structure_to_recapitulate,
        sequence_from_structure=True
    )
    problem.add_fixed_length_segment(length=30)

    start = time.time()
    runner = rfdiffusion.RFDiffusionMotif()
    print(runner.conf)
    print("*** Initialization time", time.time() - start)

    start = time.time()
    results = runner.run(
        problem,
        num=1)
    print("*** Run time", (time.time() - start))

    assert len(results) == 1
    result = results.iloc[0]
    import ipdb ; ipdb.set_trace()

    construct = result.structure
    seq = construct.ca.getSequence()

    assert len(seq) == 62
    assert seq[20:20 + 12] == handle.select(
        "chain A and resid 22 to 33").ca.getSequence()

    assert construct.select("constrained_by_sequence").ca.getSequence() == (
            structure_to_recapitulate.ca.getSequence())
    assert construct.select("constrained_by_structure").ca.getSequence() == (
            structure_to_recapitulate.ca.getSequence())
    assert len(construct.select("unconstrained").ca) == 50
