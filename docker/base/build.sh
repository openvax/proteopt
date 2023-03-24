set -e
set -x

time docker build -t timodonnell/proteopt-base-gpu:latest .
time docker build --target shrunk -t timodonnell/proteopt-base-gpu-shrunk:latest .

