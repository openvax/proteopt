set -e
set -x

time docker build --target minimal -t timodonnell/proteopt-minimal:latest .
time docker build --target base -t timodonnell/proteopt-base:latest .
time docker build -t timodonnell/proteopt-complete:latest .

