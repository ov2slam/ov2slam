OV2SLAM can be build for the use in docker.
The goal of docker build is to create a reproducible experimental environment.

## Implementation notes

- The build consists of 2 stages. First we build a docker image that contains basic dependencies, like ROS and OpenCV.
  Then we use that image to create a ROS workspace and build Thirdparty and OV2Slam itself.

- OpenCV version must match the version used in ROS. Otherwise one has to rebuild standard packages like ros-${DISTRO}-image-transport.

- OpenCV is build from sources there to enable opencv-non-free features for feature matching.

## Building


For ROS Noetic:

    cd ov2slam
    # Building an image with dependencies...
    docker build docker/ -f docker/Dockerfile -t ov2slam-noetic-dependencies --build-arg ROS_DISTRO=noetic --build-arg OPENCV_VERSION=4.2.0
    # Building SLAM image...
    docker build . -t ov2slam-noetic --build-arg ROS_DISTRO=noetic -f docker/Dockerfile.slam

For ROS Melodic:

    cd ov2slam
    # Building an image with dependencies...
    docker build docker/ -f docker/Dockerfile -t ov2slam-melodic-dependencies --build-arg ROS_DISTRO=melodic --build-arg OPENCV_VERSION=3.2.0
    # Building SLAM image...
    docker build . -t ov2slam-melodic --build-arg ROS_DISTRO=melodic -f docker/Dockerfile.slam

    

## How to run 

Below are some sample commands for ROS Noetic.

    docker run -it --rm  ov2slam-noetic bash
    # inside docker container:
    source /ws/devel/setup.bash; rosrun ov2slam ov2slam_node

Running with X forwarding (there are security concerns, see [http://wiki.ros.org/docker/Tutorials/GUI](http://wiki.ros.org/docker/Tutorials/GUI) for details)

    xhost +local:docker
    docker run -it --rm --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" ov2slam-noetic bash

For Nvidia cards:

    xhost +local:docker
    docker run -it --rm --privileged --net=host --env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all --env=DISPLAY --env=QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 ov2slam-noetic bash