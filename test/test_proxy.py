import warnings

import numpy.testing

warnings.filterwarnings("ignore")

import proteopt
import proteopt.client
import proteopt.mock_tool

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
