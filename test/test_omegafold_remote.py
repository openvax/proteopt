import warnings

warnings.filterwarnings("ignore")

import numpy

import proteopt
import proteopt.client
import proteopt.omegafold

from .util import multiple_running_server_endpoints


def test_basic_real(multiple_running_server_endpoints):
    client = proteopt.client.Client(endpoints=multiple_running_server_endpoints)
    model = client.remote_model(
        proteopt.omegafold.OmegaFold,
        max_length=16,
        num_recycle=0,
        amber_relax=False)

    items = []
    for i in range(100):
        items.append("SIIN" * numpy.random.randint(3,10))
    print(items)

    predictions = model.run_multiple(items, items_per_request=2)

    for (i, prediction) in enumerate(predictions):
        assert prediction.ca.getSequence() == items[i]
