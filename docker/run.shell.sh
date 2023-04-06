#!/bin/bash

docker run -it \
  -v "$(realpath ~):/root/host" \
  -v "$(realpath ..):/root/proteopt" \
  -p 8888:8888 \
   --gpus all \
   timodonnell/proteopt-complete:latest \
   bash -c "cd /root/proteopt/docker ; bash update.sh ; cd ; bash"
