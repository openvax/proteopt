import warnings

import numpy.testing

warnings.filterwarnings("ignore")

import proteopt
import proteopt.client
import proteopt.mock_tool
import proteopt.alphafold

from .util import running_proxy_endpoint


def test_basic(running_proxy_endpoint):
    client = proteopt.client.Client(endpoints=[running_proxy_endpoint])
    model = client.remote_model(proteopt.mock_tool.MockTool, greeting="hi")
    results = model.run_multiple([
        dict(name="tim", sleep_time=0.0),
        dict(name="joe", sleep_time=0.0, array=[1,2]),
    ])
    results = list(results)
    assert results == ["test-server: hi tim", "test-server: hi joe 3.00"]


def test_af2(running_proxy_endpoint):
    client = proteopt.client.Client(endpoints=[running_proxy_endpoint])
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


def test_af2_real(running_proxy_endpoint):
    client = proteopt.client.Client(endpoints=[running_proxy_endpoint])
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