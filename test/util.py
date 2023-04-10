import subprocess
import os
import time

import pytest

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
REPO_ROOT_DIR = os.path.abspath(os.path.join(DATA_DIR, "..", ".."))
ALPHAFOLD_WEIGHTS_DIR = "/data/static/alphafold-params/"
OMEGAFOLD_WEIGHTS_DIR = "/data/static/omegafold_ckpt/"


def run_server(port, sleep_seconds=3.0):
    endpoint_file = "/tmp/proteopt_endpoint.txt"
    try:
        os.unlink(endpoint_file)
    except IOError:
        pass
    process = subprocess.Popen(
        [
            "python",
            os.path.join(REPO_ROOT_DIR, "api.py"),
            "--debug",
            "--port", str(port),
            "--mock-server-name", 'test-server',
            "--alphafold-data-dir", ALPHAFOLD_WEIGHTS_DIR,
            "--omegafold-data-dir", OMEGAFOLD_WEIGHTS_DIR,
            "--write-endpoint-to-file", endpoint_file,
        ])
    time.sleep(sleep_seconds)
    with open(endpoint_file) as fd:
        endpoint = fd.read().strip()
    os.unlink(endpoint_file)
    return (process, endpoint + "/tool")

@pytest.fixture
def running_server_endpoint(port=0, sleep_seconds=0):
    (process, endpoint) = run_server(port)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    yield endpoint
    process.terminate()

@pytest.fixture
def multiple_running_server_endpoints(ports=(0, 0), sleep_seconds=0):
    processes = []
    endpoints = []
    for port in ports:
        (process, endpoint) = run_server(port)
        processes.append(process)
        endpoints.append(endpoint)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    yield endpoints
    for process in processes:
        process.terminate()