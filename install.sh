#! /bin/bash

PROJ_ROOT="$( cd "$( dirname "$0" )" && pwd -P )"
echo $PROJ_ROOT

cd ${PROJ_ROOT}
if [ ! -d "./FoundationPose" ]; then
    git clone https://github.com/NVlabs/FoundationPose.git
fi

#if [ ! -d "./FoundationPose/demo_data" ]; then
    #mkdir ./FoundationPose/demo_data
    #python -m pip install numpy pillow scipy shapely trimesh
    #python ycb_downloader.py
#fi

#build dockerFile
docker build -t foundationpose:webserver -f ~/pose_inferencer/docker/foundationpose.dockerFile .
