OV2SLAM can be build for the use in docker.
OpenCV version must match the version used in ROS. Otherwise one has to rebuild standard packages like `ros-${DISTRO}-image-transport`.

Below one can find commands to build docker images. OpenCV is build from sources there to enable opencv-non-free features 
for feature matching.


First build an image with dependencies

For ros-noetic:

    docker build docker/ -f docker/Dockerfile -t ov2slam-noetic-dependencies --build-arg ROS_DISTRO=noetic --build-arg OPENCV_VERSION=4.2.0


For ros-melodic TODO (not implemented):

    docker build docker/ -f docker/Dockerfile -t ov2slam-melodic-dependencies --build-arg ROS_DISTRO=melodic --build-arg OPENCV_VERSION=???


Now build the slam image:

    docker build . -t ov2slam --build-arg ROS_DISTRO=noetic -f docker/Dockerfile.slam

