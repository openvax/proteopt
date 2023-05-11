import json
from queue import Queue
import threading

import logging
import urllib3
import requests

from .remote_model import RemoteModel


class Client():
    def __init__(self, endpoints, max_retries=2):
        self.endpoints = endpoints
        self.work_queue = Queue()
        self.max_retries = max_retries

        self.threads = []
        for endpoint in endpoints:
            thread = threading.Thread(
                target=self.worker_thread,
                name="thread_%s" % endpoint,
                daemon=True,
                args=(endpoint,))
            thread.start()

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
                        except urllib3.exceptions.HTTPError as e:
                            logging.warning(
                                "HTTPError [attempt %d of %d]" % (
                                    i + 1, self.max_retries),
                                full_endpoint,
                                e)
                            session = requests.Session()
                            exception = e
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

