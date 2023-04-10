import warnings
warnings.filterwarnings("ignore")

import os
import numpy
import pandas

import prody
import yabul

import proteopt
import proteopt.alphafold

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

from .util import ALPHAFOLD_WEIGHTS_DIR


def test_basic():
    model = proteopt.alphafold.AlphaFold(
        data_dir=ALPHAFOLD_WEIGHTS_DIR,
        max_length=16,
        num_recycle=0,
        amber_relax=False)

    prediction = model.run("SIINFEKL")
    print(prediction)
    assert prediction.ca.getSequence() == "SIINFEKL"
    assert (prediction.getCoords()**2).sum() > 0
    assert prediction.getData("af2_ptm").mean() > 0


def test_prediction_with_template():
    template = prody.parsePDB(os.path.join(DATA_DIR, "1MBN.pdb"))
    template = template.select("chain A and resid 1 to 150 and resid != 100")

    target = prody.parsePDB(os.path.join(DATA_DIR, "1MBN.pdb"))
    target = target.select("protein chain A and resid 1 to 150")

    sequence = target.ca.getSequence()
    print("Target", len(target.ca), sequence)
    print("Template", len(template.ca), template.ca.getSequence())

    alignment = yabul.align_pair(
        sequence,
        template.ca.getSequence())
    print(alignment)
    print("Sequence gaps", sum(c == '-' for c in alignment.query))
    print("Template gaps", sum(c == '-' for c in alignment.reference))

    model_name = "model_1_ptm"
    num_recycle = 0
    model = proteopt.alphafold.AlphaFold(
        data_dir=ALPHAFOLD_WEIGHTS_DIR,
        max_length=len(target.ca),
        num_recycle=num_recycle,
        model_name=model_name,
        amber_relax=False)

    prediction = model.run(
        sequence,
        template=template,
        template_replace_sequence_with_gaps=False,
        template_mask_sidechains=False)

    prody.calcTransformation(prediction.ca, target.ca).apply(prediction)
    rmsd_ca = prody.calcRMSD(prediction.ca, target.ca)
    ptm, = numpy.unique(prediction.getData("af2_ptm"))
    print(
        model_name,
        "recycles=",
        num_recycle,
        "rmsd=",
        rmsd_ca,
        "ptm=",
        ptm)

    assert rmsd_ca < 2.0

