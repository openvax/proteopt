import os
import numpy
import collections
import pandas

import prody
import yabul

import proteopt
from proteopt import (
    proteinmpnn, rfdesign_inpainting, rfdesign_hallucination, alphafold)
from proteopt.scaffold_problem import ScaffoldProblem
from proteopt.common import calculate_rmsd


from .util import ALPHAFOLD_WEIGHTS_DIR, DATA_DIR

DesignResult = collections.namedtuple(
    "DesignResult", ["structure", "rmsd_ca", "ptm"])


def alphafold_predict(alphafold_runner, sequence, target_structure):
    prediction = alphafold_runner.run(
        sequence,
        template=target_structure,
        template_replace_sequence_with_gaps=True,
        template_mask_sidechains=False)
    rmsd_ca = calculate_rmsd(prediction.ca, target_structure.ca)
    ptm, = numpy.unique(prediction.getData("af2_ptm"))
    return DesignResult(prediction, rmsd_ca, ptm)


def test_basic():
    region1 = prody.parsePDB(
        os.path.join(DATA_DIR, "1MBN.pdb")
    ).select("protein chain A and resid 40 to 60")

    alphafold_runner = alphafold.AlphaFold(
        ALPHAFOLD_WEIGHTS_DIR,
        max_length=len(region1.ca),
        model_name="model_2_ptm",
        num_recycle=0,
        amber_relax=False)

    proteinpnn_runner = proteinmpnn.ProteinMPNN()
    inpainting_runner = rfdesign_inpainting.RFDesignInpainting()
    hallucination_runner = rfdesign_hallucination.RFDesignHallucination()

    native_prediction = alphafold_predict(
        alphafold_runner, region1.ca.getSequence(), region1)
    print("Native design prediction", native_prediction)

    subregion = region1.select("resid 45 to 50 or resid 54 to 58")
    subregion_prediction = alphafold_predict(
        alphafold_runner, subregion.ca.getSequence(), subregion)
    print("Subregion", subregion_prediction)

    inpainting_problem = ScaffoldProblem(subregion)
    inpainting_problem.add_segment(structure=subregion.select("resid 45 to 50"))
    inpainting_problem.add_variable_length_segment(3, 15)
    inpainting_problem.add_segment(structure=subregion.select("resid 54 to 58"))

    inpainting_results = inpainting_runner.run(inpainting_problem, num=10)

    # Next step: write some integration tests using these tools + AF2
    # Then make a real run using these tools


