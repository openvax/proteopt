import warnings

import numpy.testing

warnings.filterwarnings("ignore")

import os
import prody

import pytest

import proteopt
import proteopt.client
import proteopt.proteinmpnn

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

from .util import running_server_endpoint

def test_basic(running_server_endpoint):
    client = proteopt.client.Client(endpoints=[running_server_endpoint])
    runner = client.remote_model(
        proteopt.proteinmpnn.ProteinMPNN)

    region1 = prody.parsePDB(
        os.path.join(DATA_DIR, "1MBN.pdb")
    ).select("protein chain A and resid 10 to 39")
    sequence = region1.ca.getSequence()
    fixed_region = region1.select("resid 25 to 28 or resid 35")

    results = runner.run(region1, num=5, fixed=fixed_region)
    print(results)

    assert results.shape[0] == 5
    assert list(results.seq.str.len().unique()) == [len(region1.ca)]

    for _, row in results.iterrows():
        assert row.seq[:10] != sequence[:10]
        assert row.seq[15:19] == sequence[15:19]
        assert row.seq[25] == sequence[25]

