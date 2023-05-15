import os
import time
import proteopt
import prody
import numpy
import pickle

from proteopt.scaffold_problem import ScaffoldProblem

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


def test_pickle():
    handle = prody.parsePDB(
        os.path.join(DATA_DIR, "1MBN.pdb"), model=1)

    problem = (
        ScaffoldProblem(handle)
        .add_segment(10)
        .add_segment(
            structure=handle.select("chain A and resid 35 to 137"))
        .add_segment(10))

    problem2 = pickle.loads(pickle.dumps(problem))
    assert problem == problem2


def test_basic():
    handle = prody.parsePDB(
        os.path.join(DATA_DIR, "6JJP.pdb"), model=1)

    problem = (
        ScaffoldProblem(handle)
        .add_segment(10)
        .add_segment(
            structure=handle.select("chain C and resid 57 to 65"),
            motif_num=0)
        .add_segment(5)
        .add_segment(
            structure=handle.select("chain C and resid 70 to 75"),
            motif_num=0)
        .add_segment(10))

    fake_solution = handle.select("chain C and resid 47 to 75").copy()
    fake_solution.setCoords(fake_solution.getCoords() + 800)
    evaluation = problem.evaluate_solution(fake_solution, prefix="fake_")
    print(evaluation)
    assert evaluation["fake_motif_0_ca_rmsd"] < 1e-4
    assert evaluation["fake_motif_0_all_atom_rmsd"] < 1e-4
