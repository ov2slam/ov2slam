Docker image for a seamless and reproducible usage of OV²SLAM algorithm.

## Building

For ROS2 Humble on amd64 architecture:

```shell
cd ov2slam

# Building SLAM image with dependencies...
docker build . -f docker/Dockerfile -t ov2slam-humble-amd64 --build-arg ROS_DISTRO=humble --build-arg ARCHITECTURE=amd64
```

## Run 

```shell
docker run -it --rm  ov2slam-humble-amd64 bash

# inside docker container:
source /ws/install/setup.bash

ros2 run ov2slam ov2slam_node <CONFIG FILE>.yaml
```

In the other container run a bag with a command:
```shell
ros2 bag play <PATH TO ROS2 BAG FILE>
```

### X forwarding on Nvidia cards

```shell
xhost +local:docker

docker run -it --rm --privileged --net=host --env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all --env=DISPLAY --env=QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix -e NVIDIA_VISIBLE_DEVICES=0 ov2slam-humble-amd64 bash
```
