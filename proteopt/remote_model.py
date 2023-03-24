import pandas
from queue import Queue

from .common import serialize, deserialize


class RemoteModel():
    def __init__(self, client, tool_class, model_kwargs):
        self.client = client
        self.tool_class = tool_class
        self.base_payload = dict(model_kwargs)
        self.base_payload['tool_name'] = tool_class.tool_name
        self.most_recent_results = None

    # list_of_dicts can also be a list of tuples or a list of objects
    # these are treated as positional arguments
    def run_multiple(self, list_of_dicts, items_per_request=1, show_progress=False):
        payloads = []
        payload = None

        run_args_spec = self.tool_class.run_args
        example_arg = list(run_args_spec)[0]

        for d in list_of_dicts:
            if not isinstance(d, dict):
                if not isinstance(d, tuple):
                    d = (d,)
                    d = self.positional_paramaters_to_named_parameters(d)

            unsupported_arguments = [x for x in d if x not in run_args_spec]
            if unsupported_arguments:
                raise ValueError(
                    "Unsupported arguments for %s run: %s. Supported args: %s" % (
                        str(self.tool_class),
                        ", ".join(unsupported_arguments),
                        ", ".join(run_args_spec)))

            if payload and len(payload[example_arg]) >= items_per_request:
                payloads.append(payload)
                payload = None
            if payload is None:
                payload = dict(self.base_payload)
                for arg in run_args_spec.keys():
                    payload[arg] = []
            for arg, info in run_args_spec.items():
                try:
                    value = d[arg]
                except KeyError:
                    try:
                        value = info['default']
                    except KeyError:
                        raise ValueError("Missing required argument: %s" % arg)
                payload[arg].append(value)

        if payload:
            payloads.append(payload)

        for arg, info in run_args_spec.items():
            if info['type'] is object:
                for payload in payloads:
                    payload[arg] = serialize(payload[arg])
        try:
            result_queue = Queue()
            result_payloads = {}
            for (payload_id, payload) in enumerate(payloads):
                self.client.work_queue.put((payload_id, payload, result_queue))

            iterator = range(len(payloads))
            if show_progress:
                import tqdm
                iterator = tqdm.tqdm(iterator)

            for _ in iterator:
                (payload_id, return_payload) = result_queue.get()
                if return_payload.get("success"):
                    objs = deserialize(return_payload.pop('results'))
                    return_payload['parsed_results'] = objs
                    return_payload["payload_id"] = payload_id
                    result_payloads[payload_id] = return_payload
                elif return_payload.get("success") is False:
                    (kind, message) = return_payload["exception"]
                    if kind == "ValueError":
                        raise ValueError("Remote error: %s" % message)
                    raise RuntimeError("Remote error (%s): %s" % (kind, message))
                else:
                    raise ValueError(
                        "Remote error: %s. Request was: %s." % (
                            str(return_payload), str(payloads[payload_id])))

            return_payloads = [
                result_payloads.pop(i)
                for i in range(len(payloads))
            ]
            assert len(result_payloads) == 0

            self.most_recent_results = pandas.DataFrame(return_payloads)
            return self.most_recent_results["parsed_results"].explode().values

        finally:
            for payload in payloads:
                payload["cancelled"] = True

    def run(self, *args, **kwargs):
        kwargs.update(self.positional_paramaters_to_named_parameters(args))
        result = self.run_multiple([kwargs])
        (obj,) = result
        return obj

    def positional_paramaters_to_named_parameters(self, args):
        # Convert placement args to kwargs
        kwargs = {}
        arg_order = list(self.tool_class.run_args)
        if len(args) > len(arg_order):
            raise TypeError("Too many arguments [%d]. Args are: %s" % (
                len(args), arg_order))
        for (i, value) in enumerate(args):
            kwargs[arg_order[i]] = value
        return kwargs

    def shutdown(self):
        if self.client is not None:
            self.client.shutdown()
            self.client = None

    # For backwards compatability
    predict = run
    predict_multiple = run_multiple
