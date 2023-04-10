#!/bin/bash

docker run -it \
  -v "$(realpath ~):/root/host" \
  -v "$(realpath ..):/root/proteopt" \
   --gpus all \
   timodonnell/proteopt-complete:latest \
   bash -c "cd /root/proteopt/docker ; bash update.sh ; cd ; bash"
