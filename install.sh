#! /bin/bash

PROJ_ROOT=$( cd -- "$ (dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

cd ${PROJ_ROOT}
git clone https://github.com/NVlabs/FoundationPose.git

#build dockerFile
docker build -t foundationpose:webserver -f foundationpose.dockerFile .

