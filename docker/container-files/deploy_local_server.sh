set -e
set -x 

NUM="${1:--1}"
PORT="${2:-0}"

ALPHAFOLD_WEIGHTS_DIR="/data/static/alphafold-params/"
OMEGAFOLD_WEIGHTS_DIR="/data/static/omegafold_ckpt/"
RFDIFFUSION_WEIGHTS_DIR="/data/static/rfdiffusion-params"

ENDPOINTS_FILE=/tmp/PROTEOPT_ENDPOINTS.TXT

rm -f /tmp/proteopt_endpoint.txt
python \
    ~/proteopt/proxy.py \
    --debug \
    --port $PORT \
    --launch-servers $NUM \
    --write-endpoint-to-file $ENDPOINTS_FILE \
    --launch-args \
    --debug \
    --alphafold-data-dir "$ALPHAFOLD_WEIGHTS_DIR" \
    --omegafold-data-dir "$OMEGAFOLD_WEIGHTS_DIR" \
    --rfdiffusion-motif-models-dir "$RFDIFFUSION_WEIGHTS_DIR" \
