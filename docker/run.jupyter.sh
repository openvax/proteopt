#!/bin/bash

docker run -it \
  -v "$(realpath ~):/root/host" \
  -v "$(realpath ..):/root/proteopt" \
  -p 8888:8888 \
   --gpus all \
   timodonnell/proteopt-complete:latest \
   bash -c "cd /root/proteopt/docker ; bash update.sh ; cd ; /root/miniconda3/envs/design-env/bin/jupyter lab --ip=0.0.0.0 --allow-root"
