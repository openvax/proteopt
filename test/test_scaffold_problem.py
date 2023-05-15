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
        .add_segment(4)
        .add_segment(
            structure=handle.select("chain C and resid 70 to 75"),
            motif_num=0)
        .add_segment(11))

    fake_solution = handle.select("chain C and resid 47 to 86").copy()
    problem.check_solution_sequence_is_valid(fake_solution.ca.getSequence())
    fake_solution.setCoords(fake_solution.getCoords() + 800)
    evaluation = problem.evaluate_solution(fake_solution, prefix="fake_")
    print(evaluation)
    assert evaluation["fake_motif_0_ca_rmsd"] < 1e-4
    assert evaluation["fake_motif_0_all_atom_rmsd"] < 1e-4

def test_two_motifs():
    handle = prody.parsePDB(
        os.path.join(DATA_DIR, "6JJP.pdb"), model=1)

    problem = (
        ScaffoldProblem(handle)
        .add_segment(10)
        .add_segment(
            structure=handle.select("chain C and resid 57 to 65"),
            motif_num=0)
        .add_segment(4)
        .add_segment(
            structure=handle.select("chain C and resid 70 to 75"),
            motif_num=0)
        .add_segment(11)
        .add_segment(10)
        .add_segment(
            structure=handle.select("chain C and resid 57 to 65"),
            motif_num=1)
        .add_segment(4)
        .add_segment(
            structure=handle.select("chain C and resid 70 to 75"),
            motif_num=1)
        .add_segment(11))

    fake_solution = handle.select("chain C and resid 47 to 86").copy()
    part2 = handle.select("chain C and resid 47 to 86").copy()
    part2.setChids(["Z"] * len(part2.getChids()))
    fake_solution += part2

    problem.check_solution_sequence_is_valid(fake_solution.ca.getSequence())
    fake_solution.setCoords(fake_solution.getCoords() + 800)
    evaluation = problem.evaluate_solution(fake_solution, prefix="fake_")
    print(evaluation)
    assert evaluation["fake_motif_0_ca_rmsd"] < 1e-4
    assert evaluation["fake_motif_0_all_atom_rmsd"] < 1e-4
