ARG ROS_DISTRO=noetic
ARG BASE_IMAGE=osrf/ros:${ROS_DISTRO}-desktop-full
FROM $BASE_IMAGE

# opencv version must be consistent with the version of ROS, 
# othervize you may need to rebuild image-transport and other ROS packages
ARG OPENCV_VERSION=4.2.0


# set to true if you want to keep opencv and other sources and build files left in the docker
ARG KEEP_SOURCES=false


RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ros-${ROS_DISTRO}-pcl-ros ros-${ROS_DISTRO}-image-transport \
    libgflags-dev libgoogle-glog-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

ADD install_scripts/install_opengv.sh /install_scripts/
RUN sh /install_scripts/install_opengv.sh "/build_dep" 91f4b19c73450833a40e463ad3648aae80b3a7f3 \
    && ($KEEP_SOURCES || rm -r /build_dep)

ADD install_scripts/install_opencv_dependencies.sh install_scripts/install_opencv.sh /install_scripts/    
RUN apt-get update \ 
    && DEBIAN_FRONTEND=noninteractive sh /install_scripts/install_opencv_dependencies.sh \
    && sh /install_scripts/install_opencv.sh "/build_dep" "$OPENCV_VERSION" \
    && ($KEEP_SOURCES || rm -r /build_dep) \
    && rm -rf /var/lib/apt/lists/*