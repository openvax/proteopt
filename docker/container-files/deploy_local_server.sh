set -e
set -x 

NUM="${1:-2}"
NUM_PER=1

source ../paths.sh

ENDPOINTS_FILE=/tmp/PROTEOPT_ENDPOINTS.TXT
PIDS_FILE=/tmp/PROTEOPT_ENDPOINTS.PIDS.TXT

rm -f "$ENDPOINTS_FILE"
rm -f "$PIDS_FILE"

pids=()
for i in $(seq $NUM)
do
	for j in $(seq $NUM_PER)
	do
		CUDA_VISIBLE_DEVICES=$(expr $i - 1) python \
			~/git/proteopt/api.py \
			--debug \
			--alphafold-data-dir "$ALPHAFOLD_WEIGHTS_DIR" \
			--write-endpoint-to-file /tmp/proteopt_endpoint.txt &
		pids+=("$!")
		sleep 1
		cat /tmp/proteopt_endpoint.txt >> "$ENDPOINTS_FILE"
	done
done

echo "Endpoints:"
cat "$ENDPOINTS_FILE"

list_descendants ()
{
  local children=$(ps -o pid= --ppid "$1")

  for pid in $children
  do
    list_descendants "$pid"
  done

  echo "$children"
}
list_descendants $$ | sort | uniq | grep "\S" --color=none | grep -v $$ > "$PIDS_FILE"

trap 'kill $(cat $PIDS_FILE | xargs)' INT
wait