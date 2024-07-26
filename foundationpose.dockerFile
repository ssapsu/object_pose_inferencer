FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install ROS2 Humble

ENV TZ=ASIA/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install locales

ARG DEBIAN_FRONTEND=noninteractive
RUN locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
RUN locale
RUN apt install software-properties-common -y && add-apt-repository universe

RUN apt update && apt install curl -y
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt update -y && apt upgrade -y

RUN apt install ros-humble-desktop ros-humble-ros-base ros-dev-tools ros-humble-vision-msgs -y

# Install foundaionpose dependencies

RUN apt-get update --fix-missing && \
    apt-get install -y libgtk2.0-dev && \
    apt-get install -y wget bzip2 ca-certificates curl git vim tmux g++ gcc build-essential cmake checkinstall \
    gfortran libjpeg8-dev libtiff5-dev pkg-config yasm libavcodec-dev libavformat-dev libswscale-dev libdc1394-dev \
    libxine2-dev libv4l-dev qtbase5-dev qt5-qmake libgtk2.0-dev libtbb-dev libatlas-base-dev libfaac-dev libmp3lame-dev libtheora-dev \
    libvorbis-dev libxvidcore-dev libopencore-amrnb-dev libopencore-amrwb-dev x264 v4l-utils libprotobuf-dev protobuf-compiler \
    libgoogle-glog-dev libgflags-dev libgphoto2-dev libhdf5-dev doxygen libflann-dev libboost-all-dev proj-data libproj-dev \
    libyaml-cpp-dev cmake-curses-gui libzmq3-dev freeglut3-dev

RUN cd / && git clone https://github.com/pybind/pybind11 &&\
    cd pybind11 && git checkout v2.10.0 &&\
    mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF &&\
    make -j6 && make install

RUN cd / && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz &&\
    tar xvzf ./eigen-3.4.0.tar.gz &&\
    cd eigen-3.4.0 &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make install

RUN apt install python3-pip python-is-python3 -y && pip install torchvision==0.16.0+cu121 torchaudio==2.1.0 torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121 &&\
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" &&\
    pip install scipy joblib scikit-learn ruamel.yaml trimesh pyyaml opencv-python imageio open3d transformations warp-lang einops kornia pyrender

# RUN cd / && git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
# RUN cd /kaolin && FORCE_CUDA=1 python setup.py develop

# replace above one with following
RUN python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

RUN cd / && git clone https://github.com/NVlabs/nvdiffrast &&\
    cd /nvdiffrast && pip install .

ENV OPENCV_IO_ENABLE_OPENEXR=1

RUN pip install scikit-image meshcat webdataset omegaconf pypng roma seaborn opencv-contrib-python openpyxl \
    wandb imgaug Ninja xlsxwriter timm albumentations xatlas rtree nodejs jupyterlab objaverse g4f ultralytics==8.0.120 pycocotools videoio numba &&\
    pip install h5py

WORKDIR /ros2_ws
RUN mkdir -p /ros2_ws/src
COPY ./foundationpose /ros2_ws/src

ENV SHELL=/bin/bash
RUN ln -sf /bin/bash /bin/sh

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
