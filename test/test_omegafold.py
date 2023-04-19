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

model = proteopt.omegafold.OmegaFold(data_dir=OMEGAFOLD_WEIGHTS_DIR)

def test_basic():
    # Predict a peptide
    prediction = model.run("SIINFEKL")
    print(prediction)
    assert prediction.ca.getSequence() == "SIINFEKL"
    assert (prediction.getCoords()**2).sum() > 0

def test_multiple():
    # Run a bunch of peptide predictions at once
    items = []
    for i in range(10):
        items.append("SIIN" * numpy.random.randint(3, 10))

    predictions = model.run_multiple(items)
    for (i, prediction) in enumerate(predictions):
        assert prediction.ca.getSequence() == items[i]

def test_ground_truth():
    # Compare to ground truth for a real structure
    truth = prody.parsePDB(
        os.path.join(DATA_DIR, "1MBN.pdb")).select("protein chain A")

    prediction = model.run(truth.ca.getSequence())

    rmsd = proteopt.common.calculate_rmsd(truth, prediction)
    print("All atom rmsd", rmsd)
    assert rmsd < 3.0


