#! /bin/bash

PROJ_ROOT="$( cd "$( dirname "$0" )" && pwd -P )"
echo $PROJ_ROOT

cd ${PROJ_ROOT}
if [ ! -d "./FoundationPose" ]; then
    git clone https://github.com/NVlabs/FoundationPose.git
fi

if [ ! -d "./FoundationPose/demo_data" ]; then
    mkdir ./FoundationPose/demo_data
    python -m pip install numpy pillow scipy shapely trimesh
    python ycb_downloader.py
fi

REQ_URL="https://api.ngc.nvidia.com/v2/resources/$NGC_ORG/$NGC_TEAM/$NGC_RESOURCE/versions/$NGC_VERSION/files/$NGC_FILENAME"

mkdir -p $PROJ_ROOT/FoundationPose/weight && \
    curl -LO --request GET "${REQ_URL}" && \
    tar -xf ${NGC_FILENAME} -C $PROJ_ROOT/FoundationPose/weight && \
    rm ${NGC_FILENAME}

wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.0/zip -O foundationpose_1.0.0.zip
unzip foundationpose_1.0.0.zip -d ./FoundationPose/weight
rm foundationpose_1.0.0.zip

cp ./config.yml ./FoundationPose/weights/2024-01-11-20-02-45/config.yaml

#build dockerFile
docker build -t foundationpose:webserver -f ~/pose_inferencer/docker/foundationpose.dockerFile .
