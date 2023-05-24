import warnings

import numpy.testing

warnings.filterwarnings("ignore")

import subprocess
import os
import time

import pytest

import proteopt
import proteopt.client
import proteopt.alphafold

from .util import running_server_endpoint, multiple_running_server_endpoints


def test_multiple_endpoints_mock(multiple_running_server_endpoints):
    client = proteopt.client.Client(endpoints=multiple_running_server_endpoints)
    model = client.remote_model(
        proteopt.alphafold.AlphaFold,
        model_name="MOCK",
        max_length=16,
        num_recycle=0,
        amber_relax=False)

    predictions = list(
        model.predict_multiple(
            ["MOCKMOCK1", "MOCKMOCK2", "MOCKMOCK3", "MOCKMOCK4"]))
    assert model.most_recent_results.payload_id.nunique() == 4
    print(predictions)
    for prediction in predictions:
        assert prediction.ca.getSequence() == "AY"
        assert (prediction.getCoords()**2).sum() > 0

    predictions = list(
        model.predict_multiple(
            ["MOCKMOCK1", "MOCKMOCK2", "MOCKMOCK3", "MOCKMOCK4"],
            items_per_request=2))
    assert model.most_recent_results.payload_id.nunique() == 2
    print(predictions)
    for prediction in predictions:
        assert prediction.ca.getSequence() == "AY"
        assert (prediction.getCoords()**2).sum() > 0


def test_basic_mock(running_server_endpoint):
    client = proteopt.client.Client(endpoints=[running_server_endpoint])
    model = client.remote_model(
        proteopt.alphafold.AlphaFold,
        model_name="MOCK",
        max_length=16,
        num_recycle=0,
        amber_relax=False)

    prediction = model.predict("SIINFEKL")
    assert prediction.ca.getSequence() == "AY"
    assert (prediction.getCoords()**2).sum() > 0

    numpy.testing.assert_raises(ValueError, model.predict, "THROW")

    model.shutdown()


def test_basic_real(running_server_endpoint):
    client = proteopt.client.Client(endpoints=[running_server_endpoint])
    model = client.remote_model(
        proteopt.alphafold.AlphaFold,
        max_length=16,
        num_recycle=0,
        amber_relax=False)

    prediction = model.predict("SIINFEKL")
    print(prediction)
    assert "unrelaxed" in str(prediction)
    assert prediction.ca.getSequence() == "SIINFEKL"
    assert (prediction.getCoords()**2).sum() > 0
    assert prediction.getData("af2_ptm").mean() > 0

