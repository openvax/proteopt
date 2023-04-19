import warnings
warnings.filterwarnings("ignore")

import os
import numpy
import pandas

import prody
import yabul

import proteopt
import proteopt.common
import proteopt.omegafold

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

from .util import OMEGAFOLD_WEIGHTS_DIR

def test_basic():
    model = proteopt.omegafold.OmegaFold(data_dir=OMEGAFOLD_WEIGHTS_DIR)

    prediction = model.run("SIINFEKL")
    print(prediction)
    assert prediction.ca.getSequence() == "SIINFEKL"
    assert (prediction.getCoords()**2).sum() > 0

    items = []
    for i in range(100):
        items.append("SIIN" * numpy.random.randint(3, 10))

    predictions = model.run_multiple(items)
    for (i, prediction) in enumerate(predictions):
        assert prediction.ca.getSequence() == items[i]


def test_compare_to_ground_truth():
    model = proteopt.omegafold.OmegaFold(
        data_dir=OMEGAFOLD_WEIGHTS_DIR,
        model_num=1)

    truth = prody.parsePDB(
        os.path.join(DATA_DIR, "1MBN.pdb")).select("protein chain A")

    prediction = model.run(truth.ca.getSequence())

    rmsd = proteopt.common.calculate_rmsd(truth, prediction)
    print("All atom rmsd", rmsd)
    assert rmsd < 3.0


