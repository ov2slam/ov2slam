# System architecture: amd64 or arm64v8
ARG ARCHITECTURE=amd64
ARG ROS_DISTRO=humble
ARG BASE_IMAGE=${ARCHITECTURE}/ros:${ROS_DISTRO}-ros-core
FROM $BASE_IMAGE

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y curl \
    && sh -c 'echo "deb [arch=amd64,arm64] http://repo.ros2.org/ubuntu/main `lsb_release -cs` main" > /etc/apt/sources.list.d/ros2-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-colcon-common-extensions build-essential cmake pkg-config git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws/src/ov2slam/Thirdparty
RUN git clone https://gitlab.com/libeigen/eigen.git
RUN git clone https://github.com/opencv/opencv.git
RUN git clone https://github.com/opencv/opencv_contrib.git
RUN git clone https://github.com/laurentkneip/opengv.git

##############
### Eigen3 ###
##############
WORKDIR /ws/src/ov2slam/Thirdparty/eigen
RUN git checkout 3.3.7
RUN mkdir build && cd build/ \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14 \
    && make -j`nproc` install \
    && ln -s /usr/local/include/eigen3/Eigen /usr/local/include/Eigen \
    && rm -rf /usr/include/eigen3

##############
### OpenCV ###
##############
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        libgtk2.0-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        python3-dev \
        python3-numpy \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws/src/ov2slam/Thirdparty/opencv_contrib
RUN git checkout 4.5.0

WORKDIR /ws/src/ov2slam/Thirdparty/opencv
RUN git checkout 4.5.0
RUN mkdir build install && cd build \
    && cmake .. -D CMAKE_BUILD_TYPE=Release \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D OPENCV_EXTRA_MODULES_PATH="/ws/src/ov2slam/Thirdparty/opencv_contrib/modules" \
        -D BUILD_EXAMPLES=OFF \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D WITH_IPP=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_opencv_adas=OFF \
        -D BUILD_opencv_bgsegm=OFF \
        -D BUILD_opencv_bioinspired=OFF \
        -D BUILD_opencv_ccalib=OFF \
        -D BUILD_opencv_datasets=ON \
        -D BUILD_opencv_datasettools=OFF \
        -D BUILD_opencv_face=OFF \
        -D BUILD_opencv_latentsvm=OFF \
        -D BUILD_opencv_line_descriptor=OFF \
        -D BUILD_opencv_matlab=OFF \
        -D BUILD_opencv_optflow=ON \
        -D BUILD_opencv_reg=OFF \
        -D BUILD_opencv_saliency=OFF \
        -D BUILD_opencv_surface_matching=OFF \
        -D BUILD_opencv_text=OFF \
        -D BUILD_opencv_tracking=ON \
        -D BUILD_opencv_xobjdetect=OFF \
        -D BUILD_opencv_xphoto=OFF \
        -D BUILD_opencv_stereo=OFF \
        -D BUILD_opencv_hdf=OFF \
        -D BUILD_opencv_cvv=OFF \
        -D BUILD_opencv_fuzzy=OFF \
        -D BUILD_opencv_dnn=OFF \
        -D BUILD_opencv_dnn_objdetect=OFF \
        -D BUILD_opencv_dnn_superres=OFF \
        -D BUILD_opencv_dpm=OFF \
        -D BUILD_opencv_quality=OFF \
        -D BUILD_opencv_rapid=OFF \
        -D BUILD_opencv_rgbd=OFF \
        -D BUILD_opencv_sfm=OFF \
        -D BUILD_opencv_shape=ON \
        -D BUILD_opencv_stitching=OFF \
        -D BUILD_opencv_structured_light=OFF \
        -D BUILD_opencv_alphamat=OFF \
        -D BUILD_opencv_aruco=OFF \
        -D BUILD_opencv_phase_unwrapping=OFF \
        -D BUILD_opencv_photo=OFF \
        -D BUILD_opencv_gapi=OFF \
        -D BUILD_opencv_video=ON \
        -D BUILD_opencv_ml=ONN \
        -D BUILD_opencv_python2=OFF \
        -D WITH_GSTREAMER=OFF \
        -D ENABLE_PRECOMPILED_HEADERS=OFF \
    && make -j`nproc` install

##############
### OpenGV ###
##############
WORKDIR /ws/src/ov2slam/Thirdparty/opengv
RUN mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14 \
    && make -j`nproc` install

ADD ./Thirdparty /ws/src/ov2slam/Thirdparty

##############
### Sophus ###
##############
WORKDIR /ws/src/ov2slam/Thirdparty/Sophus
RUN mkdir build install && cd build/ \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14 -DBUILD_SOPHUS_EXAMPLES=OFF \
    && make -j`nproc` install

####################
### Ceres Solver ###
####################
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        libgoogle-glog-dev \
        libgflags-dev \
        libatlas-base-dev \
        libsuitesparse-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws/src/ov2slam/Thirdparty/ceres-solver
RUN mkdir build install && cd build/ \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native" -DCMAKE_CXX_STANDARD=14 -DBUILD_EXAMPLES=OFF \
    && make -j`nproc` install

################
### OBIndex2 ###
################
RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        libboost-system-dev libboost-filesystem-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws/src/ov2slam/Thirdparty/obindex2
RUN mkdir build && cd build/ \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14 \
    && make -j`nproc` install

################
### iBoW-LCD ###
################
WORKDIR /ws/src/ov2slam/Thirdparty/ibow_lcd
RUN mkdir build && cd build/ \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14 \
    && make -j`nproc` install

#################
### cv_bridge ###
#################
WORKDIR /ws
RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ros-perception/vision_opencv.git -b ${ROS_DISTRO} src/vision_opencv
RUN . /opt/ros/$ROS_DISTRO/setup.sh && colcon build --symlink-install --packages-select cv_bridge

###############
### OV²SLAM ###
###############
RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ros-${ROS_DISTRO}-image-transport \
        libpcl-dev \
        ros-${ROS_DISTRO}-pcl-ros \
        ros-${ROS_DISTRO}-pcl-msgs \
        ros-${ROS_DISTRO}-pcl-conversions \
        ros-${ROS_DISTRO}-tf2 \
        ros-${ROS_DISTRO}-tf2-geometry-msgs \
        ros-${ROS_DISTRO}-ros2bag \
        ros-${ROS_DISTRO}-rosbag2-storage-default-plugins \
        ros-${ROS_DISTRO}-rviz2 \
        ros-${ROS_DISTRO}-backward-ros \
    && rm -rf /var/lib/apt/lists/*

ADD . /ws/src/ov2slam

WORKDIR /ws
RUN . /opt/ros/$ROS_DISTRO/setup.sh && colcon build
