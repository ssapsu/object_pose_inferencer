#!/bin/bash

# .env 파일 로드
export $(grep -v '^#' .env | xargs)

PROJ_ROOT=$( cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

docker run -it --rm --gpus "device=0" -v $PROJ_ROOT/FoundationPose:/FoundationPose -v $PROJ_ROOT/webserver:/webserver -p 5678:$PORT --env NVIDIA_DISABLE_REQUIRE=1 --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -e GIT_INDEX_FILE --ipc=host --name pose_inferencer foundationpose:webserver /bin/bash
