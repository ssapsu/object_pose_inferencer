#! /bin/bash

PROJ_ROOT="$( cd "$( dirname "$0" )" && pwd -P )"
echo $PROJ_ROOT

cd ${PROJ_ROOT}
if [ ! -d "FoundationPose" ]; then
    git clone https://github.com/NVlabs/FoundationPose.git
fi

python -m pip install numpy pillow scipy shapely trimesh
python ycb_downloader.py
mkdir FoundationPose/demo_data
mv $PROJ_ROOT/models $PROJ_ROOT/FoundationPose/demo_data

# #
# mkdir FoundationPose/weight
# mv

# #build dockerFile
# docker build -t foundationpose:webserver -f ~/pose_inferencer/docker/foundationpose.dockerFile .
