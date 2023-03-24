# Mock tool for testing

import collections
import numpy
import time
from typing import Optional

from .common import args_from_function_signature


class MockTool(object):
    tool_name = "mock"

    def __init__(
            self,
            server_name : str = "mock-server",
            greeting : str = "hello"):
        self.server_name = server_name
        self.greeting = greeting

    config_args = args_from_function_signature(
        __init__, include=["server_name"])
    model_args = args_from_function_signature(
        __init__, exclude=config_args.keys())

    def run_multiple(self, list_of_dicts):
        results = []
        for kwargs in list_of_dicts:
            result = self.run(**kwargs)
            results.append(result)
        return results

    def run(
            self,
            name: str,
            sleep_time: float = 0.0,
            array: Optional[numpy.ndarray] = None,
            show_types: bool = False):
        time.sleep(sleep_time)
        result = "%s: %s %s" % (self.server_name, self.greeting, name)
        if array is not None:
            result += " " + ("%0.2f" % sum(array))

        if show_types:
            result += " %s %s %s %s" % (
                type(name), type(sleep_time), type(array), type(show_types))

        return result

    run_args = args_from_function_signature(run)
