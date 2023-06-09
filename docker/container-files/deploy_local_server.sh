set -e
set -x 

NUM="${1:--1}"
HOST="${2:-127.0.0.1}"
PORT="${3:-0}"

ALPHAFOLD_WEIGHTS_DIR="/data/static/alphafold-params/"
OMEGAFOLD_WEIGHTS_DIR="/data/static/omegafold_ckpt/"
RFDIFFUSION_WEIGHTS_DIR="/data/static/rfdiffusion-params"

ENDPOINTS_FILE=/tmp/PROTEOPT_ENDPOINTS.TXT
WORKERS_ENDPOINTS_FILE=/tmp/PROTEOPT_ENDPOINTS.WORKERS.TXT

rm -f /tmp/proteopt_endpoint.txt
python \
    ~/proteopt/proxy.py \
    --debug \
    --host $HOST \
    --port $PORT \
    "${@:4}" \
    --launch-servers $NUM \
    --write-endpoint-to-file "$ENDPOINTS_FILE" \
    --write-launched-endpoints-to-file "$WORKERS_ENDPOINTS_FILE" \
    --launch-args \
    --debug \
    --alphafold-data-dir "$ALPHAFOLD_WEIGHTS_DIR" \
    --omegafold-data-dir "$OMEGAFOLD_WEIGHTS_DIR" \
    --rfdiffusion-motif-models-dir "$RFDIFFUSION_WEIGHTS_DIR" \
    --rfdiffusion-motif-num-processes 0
