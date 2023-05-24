import argparse
import queue
import subprocess
import signal
import os
import sys
import glob
import logging
import socket
import time
import tempfile

from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource, inputs

import proteopt.client

from proteopt.common import serialize, deserialize

app = Flask(__name__)
api = Api(app)


class Proxy(Resource):
    endpoints = set()
    max_retries = None
    client = None

    @classmethod
    def get_client(cls):
        if cls.client is None:
            if not cls.endpoints:
                raise ValueError("No endpoints")
            cls.client = proteopt.client.Client(
                endpoints=[e + "/tool" for e in cls.endpoints],
                max_retries=cls.max_retries,
                extra_parallelism_factor=1)
        return cls.client

    def get(self, action, name):
        if action == "add-endpoint":
            endpoint = request.args.get('endpoint')
            self.endpoints.add(endpoint)
            self.client = None
            return f"Added endpoint {endpoint}"
        elif action == "remove-endpoint":
            endpoint = request.args.get('endpoint')
            if endpoint in self.endpoints:
                self.endpoints.remove(endpoint)
                self.client = None
                return f"Removed endpoint {endpoint}"
            else:
                return f"No such endpoint {endpoint}"
        elif action == "status":
            lines = []
            lines.extend(sorted(self.endpoints))
            return "\n".join(lines)
        elif action == "clear":
            self.endpoints.clear()
            self.client = None
            return "Cleared endpoints"
        return str(self.MODEL_CACHE.keys())

class Tool(Resource):
    def get(self, tool_name):
        max_parallelism = Proxy.get_client().max_parallelism
        result = {
            'description': 'proxy',
            'endpoints': sorted(Proxy.endpoints),
            'max_parallelism': max_parallelism,
        }
        return result, 200

    def post(self, tool_name):
        payload = request.get_json()
        payload['tool_name'] = tool_name

        client = Proxy.get_client()
        result_queue = queue.Queue()
        client.work_queue.put((0, payload, result_queue))
        (payload_id, return_payload) = result_queue.get()
        assert payload_id == 0
        return return_payload, 200


api.add_resource(Proxy, '/proxy/<action>')
api.add_resource(Tool, '/tool/<tool_name>')


# Run the test server
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--no-cleanup", action="store_true", default=False)
arg_parser.add_argument("--max-retries", default=2, type=int)
arg_parser.add_argument("--endpoints", nargs="+")
arg_parser.add_argument("--host", default="127.0.0.1")
arg_parser.add_argument("--port", type=int)
arg_parser.add_argument("--write-endpoint-to-file")
arg_parser.add_argument("--write-launched-endpoints-to-file")
arg_parser.add_argument(
    "--debug",
    default=False,
    action="store_true")

arg_parser.add_argument(
    "--launch-servers",
    metavar="N",
    type=int,
    help="Launch N API servers. If N=-K, then K servers are launched per GPU and "
    "the CUDA_VISIBLE_DEVICES parameter is set accordingly for each server.")
arg_parser.add_argument(
    "--launch-args",
    nargs=argparse.REMAINDER,
    help="All following args are args for launched API servers.")

if __name__ == '__main__':
    args = arg_parser.parse_args(sys.argv[1:])
    logging.basicConfig(level=logging.INFO)

    endpoint_to_process = {}
    work_dir = None
    if args.launch_servers:
        print(args)
        num_to_launch = args.launch_servers
        set_cuda_visible_devices = False
        num_per_gpu = None
        if args.launch_servers < 0:
            num_per_gpu = -args.launch_servers
            gpu_lines = subprocess.check_output(["nvidia-smi", "-L"]).decode().split("\n")
            gpu_lines = [g.strip() for g in gpu_lines]
            gpu_lines = [g for g in gpu_lines if g.startswith("GPU ")]
            print(f"Detected {len(gpu_lines)} GPUs.")
            num_to_launch = len(gpu_lines) * num_per_gpu
            print(f"Will launch {num_to_launch} processes on GPUs.")
            set_cuda_visible_devices = True

        work_dir = tempfile.TemporaryDirectory(prefix="proteopt_proxy_")
        for i in range(num_to_launch):
            endpoint_file = os.path.join(work_dir.name, f"endpoint.{i}.txt")
            sub_args = [
                "python",
                os.path.join(os.path.dirname(__file__), "api.py"),
            ]
            sub_args.extend(args.launch_args)
            sub_args.extend(["--write-endpoint-to-file", endpoint_file])
            environ = os.environ.copy()
            if set_cuda_visible_devices:
                environ["CUDA_VISIBLE_DEVICES"] = str(i // num_per_gpu)
            print(f"Launching API server {i} / {num_to_launch} with args:")
            print(sub_args)

            logfile = os.path.join(work_dir.name, f"log.{i}.txt")
            logfile_fd = open(logfile, "w+b")
            process = subprocess.Popen(
                sub_args, stderr=logfile_fd, stdout=logfile_fd, env=environ)
            while process.poll() is None and not os.path.exists(endpoint_file):
                time.sleep(0.1)
            try:
                endpoint = open(endpoint_file).read().strip()
            except IOError:
                print("Failed to load endpoint file. Process log:")
                logfile_fd.seek(0)
                for line in logfile_fd.readlines():
                    print(line)
                raise
            print(f"API server {i} at endpoint {endpoint} will log to {logfile}")
            endpoint_to_process[endpoint] = process
        Proxy.endpoints.update(list(endpoint_to_process))

        if args.write_launched_endpoints_to_file:
            with open(args.write_launched_endpoints_to_file, "w") as fd:
                for endpoint in Proxy.endpoints:
                    fd.write(endpoint)
                    fd.write("\n")
            print("Wrote", args.write_launched_endpoints_to_file)

    Proxy.max_retries = args.max_retries
    if args.endpoints:
        Proxy.endpoints.update(args.endpoints)

    print("Initialized proxy with endpoints: ", Proxy.endpoints)

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

    def cleanup(sig, frame):
        if args.debug:
            print("Dumping logs.")
            for g in glob.glob(os.path.join(work_dir.name, "*.txt")):
                print("*" * 40)
                print(g)
                print("*" * 40)
                for line in open(g).readlines():
                    print("---", line.rstrip())

        if work_dir is not None and not args.no_cleanup:
            print(f"Cleaning up {work_dir}")
            work_dir.cleanup()

        while endpoint_to_process:
            endpoint, process = endpoint_to_process.popitem()
            print(f"Terminating process with endpoint {endpoint}")
            process.terminate()
            if process.poll() is None:
                process.kill()
        print("Done.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    app.run(
        host=args.host,
        port=port,
        debug=args.debug,
        use_reloader=False,
        threaded=True)
