#! /bin/bash

# import port and other environment variables
export $(grep -v '^#' .env | xargs)

PROJ_ROOT=$( cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

docker run -it --rm -v $PROJ_ROOT/FoundationPose:/FoundationPose -p 5678:$PORT --name pose_inferencer foundationpose:webserver /bin/bash
