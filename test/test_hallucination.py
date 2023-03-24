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
    problem.add_fixed_length_segment(length=16)
    problem.add_fixed_length_segment(sequence="DPSK", length=4)
    problem.add_fixed_length_segment(
        structure=structure_to_recapitulate,
        sequence=structure_to_recapitulate.ca.getSequence(),
    )
    problem.add_fixed_length_segment(sequence="VTLADAGF")
    problem.add_fixed_length_segment(length=22)

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


def Xtest_custom_loss():
    handle = prody.parsePDB(os.path.join(DATA_DIR, "1MBN.pdb"), model=1)
    print(len(handle))

    structure_to_recapitulate = handle.select("chain A and resid 22 to 33")
    print(len(structure_to_recapitulate.ca), structure_to_recapitulate.ca.getSequence())

    problem = ScaffoldProblem(handle)
    problem.add_fixed_length_segment(length=16)
    problem.add_fixed_length_segment(
        structure=structure_to_recapitulate,
    )
    problem.add_fixed_length_segment(length=22)
    problem.add_fixed_length_segment(
        structure=structure_to_recapitulate,
    )
    problem.add_fixed_length_segment(length=22)
    problem.add_fixed_length_segment(
        structure=structure_to_recapitulate,
    )
    problem.add_fixed_length_segment(length=16)

    start = time.time()
    runner = rfdesign_hallucination.RFDesignHallucination()
    print("*** Initialization time", time.time() - start)

    con_hal_idx0_motif_groups = numpy.array([0] * 14 + [1] * 14 + [2] * 14)

    def add_extra_losses(ml, mappings, xyz_ref):
        def multicopy_crmsd_loss(net_out, motif_mappings):
            values = loss.superimpose_pred_xyz(net_out['xyz'], xyz_ref, motif_mappings)
            pred_centroid, ref_centroid, rot = values
            xyz_sup = (net_out['xyz'] - pred_centroid) @ rot[:,None,:,:] + ref_centroid
            return loss.calc_crd_rmsd(xyz_sup, xyz_ref, motif_mappings)

        original_con_ref_idx0 = numpy.array(mappings['con_ref_idx0'])
        original_con_hal_idx0 = numpy.array(mappings['con_hal_idx0'])

        for i in range(3):
            motif_mappings = {
                'con_ref_idx0': original_con_ref_idx0[con_hal_idx0_motif_groups == i],
                'con_hal_idx0': original_con_hal_idx0[con_hal_idx0_motif_groups == i],
            }
            ml.add(
                'mcrmsd_%d' % i,
                functools.partial(multicopy_crmsd_loss, motif_mappings=motif_mappings),
                weight=1.0)

    results = runner.run(
        problem,
        num=1,
        steps="g100,m10",
        w_crmsd=1,
        w_rog=1,
        add_extra_losses_function=add_extra_losses)
    print("*** Run time #%d" % (time.time() - start))

