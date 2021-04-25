OV2SLAM can be build for the use in docker.
The goal of docker build is to create a reproducible experimental environment. 

## Implementation notes

* The build consists of 2 stages. First we build a docker image that contains basic dependencies, like ROS and OpenCV.
Then we use that image to create a ROS workspace and build `Thirdparty` and `OV2Slam` itself.

* OpenCV version must match the version used in ROS. Otherwise one has to rebuild standard packages like `ros-${DISTRO}-image-transport`.

* OpenCV is build from sources there to enable opencv-non-free features for feature matching. 


## Building 

First build an image with dependencies.

For ros-noetic:

    docker build docker/ -f docker/Dockerfile -t ov2slam-noetic-dependencies --build-arg ROS_DISTRO=noetic --build-arg OPENCV_VERSION=4.2.0


For ros-melodic TODO (not tested):

    docker build docker/ -f docker/Dockerfile -t ov2slam-melodic-dependencies --build-arg ROS_DISTRO=melodic --build-arg OPENCV_VERSION=3.2.0


Now build SLAM image:

    docker build . -t ov2slam-noetic --build-arg ROS_DISTRO=noetic -f docker/Dockerfile.slam

How to run (with X-forwarding, intel graphics):

    docker run -it --rm  ov2slam-noetic bash
    # inside docker container:
    source /ws/devel/setup.bash
    rosrun ov2slam ov2slam_node
