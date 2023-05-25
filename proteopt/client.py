import json
import time
from queue import Queue
import threading

import logging
import urllib3
import requests

from .remote_model import RemoteModel


class Client():
    def __init__(self, endpoints, max_retries=2, extra_parallelism_factor=1):
        self.endpoints = endpoints
        self.work_queue = Queue()
        self.max_retries = max_retries
        self.threads = []

        for endpoint in endpoints:
            session = requests.Session()
            full_endpoint = endpoint + "/info"
            info = session.get(full_endpoint)
            if info.status_code != 200:
                raise IOError(f"Couldn't get info for {full_endpoint}: {info.status_code} {info.text}")
            max_parallelism = info.json()['max_parallelism'] * extra_parallelism_factor
            print(f"Client: endpoint {endpoint} will use max_parallelism {max_parallelism}")
            for i in range(max_parallelism):
                thread = threading.Thread(
                    target=self.worker_thread,
                    name=f"thread_{i}_{endpoint}",
                    daemon=True,
                    args=(endpoint,))
                self.threads.append(thread)
                thread.start()

        self.max_parallelism = max(1, len(self.threads))

    def shutdown(self):
        work_queue = self.work_queue
        self.work_queue = None
        for _ in self.threads:
            work_queue.put(None)
        for thread in self.threads:
            thread.join()

    def worker_thread(self, endpoint):
        session = requests.Session()
        while True:
            if self.work_queue is None:
                break
            tpl = self.work_queue.get()
            if tpl is not None:
                (payload_id, payload, result_queue) = tpl
                if not payload.get("cancelled"):
                    exception = None
                    result = None

                    payload_without_tool_name = dict(payload)
                    full_endpoint = endpoint + "/" + payload_without_tool_name.pop(
                        'tool_name')
                    for i in range(self.max_retries):
                        try:
                            result = session.post(
                                full_endpoint, json=payload_without_tool_name)
                        except (IOError, urllib3.exceptions.HTTPError) as e:
                            logging.warning(
                                "IOError or HTTPError [attempt %d of %d]: %s %s %s" % (
                                    i + 1, self.max_retries, full_endpoint, type(e), e))
                            session = requests.Session()
                            exception = e
                            time.sleep(2**i)
                        except TypeError as e:
                            exception = e
                            break
                    if result is not None:
                        assert result.text is not None
                        return_payload = json.loads(result.text)
                    elif exception is not None:
                        return_payload = {
                            "success": False,
                            "exception": (
                                exception.__class__.__name__, str(exception)),
                        }
                    else:
                        assert False
                    return_payload["endpoint"] = full_endpoint
                    result_queue.put((payload_id, return_payload))

    def remote_model(self, tool_class, **model_kwargs):
        return RemoteModel(self, tool_class, model_kwargs)

