import warnings

import numpy.testing

warnings.filterwarnings("ignore")

import proteopt
import proteopt.client
import proteopt.mock_tool

from .util import running_server_endpoint, multiple_running_server_endpoints


def test_multiple_endpoints_mock(multiple_running_server_endpoints):
    client = proteopt.client.Client(endpoints=multiple_running_server_endpoints)
    model = client.remote_model(proteopt.mock_tool.MockTool, greeting="hi")
    results = model.run_multiple([
        dict(name="tim", sleep_time=0.0),
        dict(name="joe", sleep_time=0.0, array=[1,2]),
    ])
    results = list(results)
    assert results == ["test-server: hi tim", "test-server: hi joe 3.00"]


def test_basic_mock(running_server_endpoint):
    client = proteopt.client.Client(endpoints=[running_server_endpoint])
    model = client.remote_model(
        proteopt.mock_tool.MockTool, greeting="hi", show_types=False)
    result = model.run(name="tim", sleep_time=0.1)
    assert result == "test-server: hi tim"

    result = model.run(name="tim", sleep_time=0.1, array=numpy.array([1,2,3]))
    assert result == "test-server: hi tim 6.00"

    result = model.run(
        name="tim", sleep_time=0.1, array=numpy.array([1, 2, 3]), show_types=True)
    assert result == (
        "test-server: hi tim 6.00 <class 'str'> <class 'float'> <class 'numpy.ndarray'> <class 'bool'>")

    numpy.testing.assert_raises(
        RuntimeError, model.run, name="tim", sleep_time=0.1, array=False)

    model = client.remote_model(proteopt.mock_tool.MockTool)
    result = model.run(name="tim", sleep_time=0.0)
    assert result == "test-server: hello tim"

    numpy.testing.assert_raises(
        ValueError, model.run, name="tim", sleep_time="foo", array=False)

    model.shutdown()
