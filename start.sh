#! /bin/bash

PROJ_ROOT=$( cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

docker run -it -v $PROJ_ROOT/FoundationPose:/FoundationPose --name pose_inferencer foundationpose:webserver /bin/bash
