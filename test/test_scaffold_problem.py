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
        .add_fixed_length_segment(10)
        .add_fixed_length_segment(
            structure=handle.select("chain A and resid 35 to 137"))
        .add_fixed_length_segment(10))

    assert list(problem.fixed_length_problems()) == [problem]

    problem2 = pickle.loads(pickle.dumps(problem))
    assert list(problem2.fixed_length_problems()) == [problem2]


def test_basic():
    handle = prody.parsePDB(
        os.path.join(DATA_DIR, "6JJP.pdb"), model=1)

    problem = (
        ScaffoldProblem(handle)
        .add_fixed_length_segment(10)
        .add_fixed_length_segment(
            structure=handle.select("chain A and resid 100 to 110"))
        .add_fixed_length_segment(10))
    assert list(problem.fixed_length_problems()) == [problem]

    problem = (
        ScaffoldProblem(handle)
        .add_fixed_length_segment(structure=handle.select("chain B and resid 35 to 137"))
        .add_variable_length_segment(10, 15)
        .add_variable_length_segment(30, 34)
        .add_fixed_length_segment(structure=handle.select("chain B and resid 208 to 212"))
        .add_fixed_length_segment(50)
        .add_fixed_length_segment(structure=handle.select("chain B and resid 200 to 201"))
        .add_variable_length_segment(100, 110))

    all_fixed_length = list(problem.fixed_length_problems())
    assert len(all_fixed_length) == 6 * 5 * 11

    def do_assertions(fixed_problem):
        assert fixed_problem.is_fixed_length()
        segments = fixed_problem.segments

        assert segments[0].length == 103
        assert 10 <= segments[1].length <= 15
        assert 30 <= segments[2].length <= 34
        assert segments[3].length == 5
        assert segments[4].length == 50
        assert segments[5].length == 2
        assert 100 <= segments[6].length <= 110

    for fixed_problem in all_fixed_length:
        do_assertions(fixed_problem)

    some_fixed_length = list(problem.fixed_length_problems(5))
    assert len(some_fixed_length) == 5
    for fixed_problem in some_fixed_length:
        do_assertions(fixed_problem)
