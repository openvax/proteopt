import argparse
import collections
import traceback
import os
import sys
import logging
import socket
import time

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource, inputs

import proteopt.alphafold
import proteopt.proteinmpnn
import proteopt.mock_tool
import proteopt.omegafold
from proteopt.common import serialize, deserialize

app = Flask(__name__)
api = Api(app)


def add_argument(parser, arg_name, info, append=False):
    type = info['type']
    if type is object:
        type = str  # We will serialize objects

    d = {
        'type': type,
    }
    if 'default' in info:
        d["default"] = info["default"]
    else:
        d["required"] = True
    if append:
        d["action"] = "append"
    parser.add_argument(arg_name, **d)


TOOL_CLASSES = [
    proteopt.mock_tool.MockTool,
    proteopt.alphafold.AlphaFold,
    proteopt.proteinmpnn.ProteinMPNN,
    proteopt.omegafold.OmegaFold,
]
TOOLS = dict((cls.tool_name, cls) for cls in TOOL_CLASSES)


class Tool(Resource):
    configuration = None  # this should be set when the app is launched

    tool_parsers = {}
    for (tool_name, tool_class) in TOOLS.items():
        tool_parsers[tool_name] = reqparse.RequestParser()
        for parameter, info in tool_class.model_args.items():
            add_argument(tool_parsers[tool_name], parameter, info)
        for parameter, info in tool_class.run_args.items():
            add_argument(
                tool_parsers[tool_name],
                parameter,
                info,
                append=not info['type'] is object)

    MODEL_CACHE = collections.OrderedDict()

    def get_model(self, tool_name, args):
        tool_class = TOOLS[tool_name]
        args_dict = dict(self.configuration[tool_name])
        cache_key = []
        for name, info in tool_class.model_args.items():
            value = getattr(args, name)
            cache_key.append((name, value))
            args_dict[name] = value

        cache_key = tuple(cache_key)

        try:
            return self.MODEL_CACHE[cache_key]
        except KeyError:
            pass

        logging.info("Loading new model: %s %s", tool_name, str(args_dict))

        model = tool_class(**args_dict)
        if len(self.MODEL_CACHE) >= self.configuration["model_cache_size"]:
            self.MODEL_CACHE.popitem(last=False)
        self.MODEL_CACHE[cache_key] = model
        return model

    def get(self, tool_name):
        return str(self.MODEL_CACHE.keys())

    def post(self, tool_name):
        tool_class = TOOLS[tool_name]

        parser = self.tool_parsers[tool_name]
        args = parser.parse_args()
        try:
            total_start = time.time()
            model = self.get_model(tool_name, args)
            init_seconds = time.time() - total_start

            run_arg_names = list(tool_class.run_args)

            for arg in run_arg_names:
                if tool_class.run_args[arg]['type'] is object:
                    setattr(args, arg, deserialize(getattr(args, arg)))

            example_run_arg = run_arg_names[0]
            list_of_input_dicts = []
            for i in range(len(getattr(args, example_run_arg))):
                d = dict((arg, getattr(args, arg)[i]) for arg in run_arg_names)
                list_of_input_dicts.append(d)

            start = time.time()
            results = model.run_multiple(list_of_input_dicts)
            assert not any(x is None for x in results)
            payload = {
                "success": True,
                "results": serialize(results),
                "init_seconds": init_seconds,
                "total_seconds": time.time() - start,
            }
            return payload, 200
        except Exception as e:
            exc_info = sys.exc_info()
            message = ''.join(traceback.format_exception(*exc_info))
            payload = {
                "success": False,
                "exception": (e.__class__.__name__, message),
            }
            return payload, 500


api.add_resource(Tool, '/tool/<tool_name>')

# Run the test server
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    "--debug",
    default=False,
    action="store_true")

arg_parser.add_argument(
    "--cuda-visible-devices")

arg_parser.add_argument("--host", default="127.0.0.1")
arg_parser.add_argument("--write-endpoint-to-file")
arg_parser.add_argument("--port", type=int)
arg_parser.add_argument("--model-cache-size", type=float, default=1.0)


arg_names_to_tool_configs = {}
for tool_name, tool_class in TOOLS.items():
    for parameter, info in tool_class.config_args.items():
        arg_name = "%s_%s" % (tool_name, parameter)
        arg_names_to_tool_configs[arg_name] = (tool_name, parameter)
        add_argument(arg_parser, "--" + arg_name.replace("_", "-"), info)

if __name__ == '__main__':
    args = arg_parser.parse_args(sys.argv[1:])

    # tool name -> dict
    tool_configs = dict((tool_name, {}) for tool_name in TOOLS.keys())
    for (arg, (tool, parameter)) in arg_names_to_tool_configs.items():
        tool_configs[tool][parameter] = getattr(args, arg)

    print("Tool configuration parameters:")
    for name, d in tool_configs.items():
        print(name)
        for (k, v) in d.items():
            print("\t%15s = %15s" % (k, v))
        print()

    Tool.configuration = dict(tool_configs)
    Tool.configuration["model_cache_size"] = args.model_cache_size

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    port = args.port
    if not port:
        # Identify an available port
        # Based on https://stackoverflow.com/questions/5085656/how-to-select-random-port-number-in-flask
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((args.host, 0))
        port = sock.getsockname()[1]
        sock.close()

    endpoint = "http://%s:%d" % (args.host, port)
    print("Endpoint will be", endpoint)
    if args.write_endpoint_to_file:
        with open(args.write_endpoint_to_file, "w") as fd:
            fd.write(endpoint)
            fd.write("\n")
        print("Wrote", args.write_endpoint_to_file)

    app.run(host=args.host, port=port, debug=args.debug, use_reloader=False)