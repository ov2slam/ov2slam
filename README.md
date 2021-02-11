# OV²SLAM
## A Fully Online and Versatile Visual SLAM for Real-Time Applications

**Paper**: [[arXiv]](https://arxiv.org/pdf/2102.04060.pdf)

**Videos**: [[video #1]](https://www.youtube.com/watch?v=N4LFD4WKHyg), [[video #2]](https://www.youtube.com/watch?v=N5O0-0339fU), [[video #3]](https://www.youtube.com/watch?v=zNevDT12cKI), [[video #4]](https://www.youtube.com/watch?v=xhLZGDdb0FU), [[video #5]](https://www.youtube.com/watch?v=ITE1yYA5B78), [[video #6]](https://www.youtube.com/watch?v=9D66qpzBvi4)

**Authors:** Maxime Ferrera, Alexandre Eudes, Julien Moras, Martial Sanfourche, Guy Le Besnerais 
(maxime.ferrera@gmail.com / first.last@onera.fr).

<img src="support_files/ov2slam_readme.gif" width = 512 height = 288 />

**OV²SLAM** is a fully real-time **Visual SLAM** algorithm for **Stereo** and **Monocular** cameras.  A complete SLAM pipeline
is implemented with a carefully designed multi-threaded architecture allowing to perform Tracking, Mapping, Bundle Adjustment and Loop Closing in real-time.
The Tracking is based on an undirect Lucas-Kanade optical-flow formulation and provides camera poses estimations at the camera's frame-rate.
The Mapping works at the keyframes' rate and ensures continuous localization by populating the sparse 3D map and minimize drift through a local map tracking step.
Bundle Adjustment is applied with an anchored inverse depth formulation, reducing the parametrization of 3D map points to 1 parameter instead of 3.
Loop Closing is performed through an **Online Bag of Words** method thanks to [iBoW-LCD](https://github.com/emiliofidalgo/ibow-lcd).  In opposition to classical
offline BoW methods, no pre-trained vocabulary tree is required.  Instead, the vocabulary tree is computed online from the descriptors extracted in the incoming video
stream, making it always suited to the currently explored environment.

## Related Paper:

If you use OV²SLAM in your work, please cite it as:


```
@article{fer2021ov2slam,
      title={{OV$^{2}$SLAM} : A Fully Online and Versatile Visual {SLAM} for Real-Time Applications},
      author={Ferrera, Maxime and Eudes, Alexandre and Moras, Julien and Sanfourche, Martial and {Le Besnerais}, Guy.},
      journal={IEEE Robotics and Automation Letters},
      year={2021}
     }
```

## License

OV²SLAM is released under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.txt). For a closed-source version of OV²SLAM for commercial purposes, please contact [ONERA](https://www.onera.fr/en/contact-us) (https://www.onera.fr/en/contact-us) or the authors. 

Copyright (C) 2020 [ONERA](https://www.onera.fr/en)

## 1. Prerequisites

The library has been tested with **Ubuntu 16.04 and 18.04**, **ROS Kinetic and Melodic** and **OpenCV 3**.  It should also work with **ROS Noetic and OpenCV 4** but this configuration has not been fully tested.

### 1.0 C++11 or Higher

OV²SLAM makes use of C++11 features and should thus be compiled with a C++11 or higher flag.

### 1.1 ROS

ROS is used for reading the video images through bag files and for visualization purpose in Rviz.

[ROS Installation](http://wiki.ros.org/ROS/Installation)

Make sure that the pcl_ros package is installed :

```
    sudo apt install ros-distro-pcl-ros
```

or even

```
    rosdep install ov2slam
```



### 1.2 Eigen3

[Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page) is used throughout OV²SLAM.  It should work with version >= 3.3.0, lower versions have not been tested.


### 1.3 OpenCV

OpenCV 3 has been used for the development of OV²SLAM, OpenCV 4 might be supported as well but it has not been tested.
(Optional) The use of BRIEF descriptor requires that **opencv_contrib** was installed.  If it is not the case, ORB will be used instead without scale and rotation invariance properties (which should be the exact equivalent of BRIEF).

**WATCH OUT** By default the CMakeLists.txt file assumes that opencv_contrib is installed, __set the OPENCV_CONTRIB flag to OFF
in CMakeLists.txt if it is not the case__.

### 1.4 iBoW-LCD

A modified version of [iBoW-LCD](https://github.com/emiliofidalgo/ibow-lcd) is included in the Thirdparty folder.  It has been turned into a shared lib and
is not a catkin package anymore.  Same goes for [OBIndex2](https://github.com/emiliofidalgo/obindex2), the required dependency for iBoW-LCD.
Check the lcdetector.h and lcdetector.cc files to see the modifications w.r.t. to the original code.

### 1.5 Sophus

[Sophus](https://github.com/strasdat/Sophus) is used for _*SE(3), SO(3)*_ elements representation.  For convenience, a copy of Sophus has been included in the Thirdparty folder.

### 1.6 Ceres Solver

[Ceres](https://github.com/ceres-solver/ceres-solver) is used for optimization related operations such as PnP, Bundle Adjustment or PoseGraph Optimization.
For convenience, a copy of Ceres has been included in the Thirdparty folder.
Note that [Ceres dependencies](http://ceres-solver.org/installation.html) are still required.

### 1.6 (Optional) OpenGV

[OpenGV](https://github.com/laurentkneip/opengv) can be used for Multi-View-Geometry (MVG) operations.  The results reported in the paper were obtained using OpenGV.
For convenience, if OpenGV is not installed, MVG operations' alternatives are proposed with OpenCV functions.  
**Note** that the performances might be lower without OpenGV.


## 2. Installation

### 2.0 Clone

Clone the git repository in your catkin workspace:

```
    cd ~/catkin_ws/src/
    git clone https://github.com/ov2slam/ov2slam.git
```

### 2.1 Build Thirdparty libs

For convenience we provide a script to build the Thirdparty libs:

```
    cd ~/catkin_ws/src/ov2slam
    chmod +x build_thirdparty.sh
    ./build_thirdparty.sh
```

**WATCH OUT** By default, the script builds obindex2, ibow-lcd, sophus and ceres.  If you want to use your own version of Sophus or Ceres 
you can comment the related lines in the script.  Yet, about Ceres, as OV²SLAM is by default compiled with the "-march=native" flag, the 
Ceres lib linked to OV²SLAM must be compiled with this flag as well, which is not the default case (at least since Ceres 2.0).  The _*build_thirdparty.sh*_ script ensures that Ceres builds with the "-march=native" flag.

If you are not interested in the Loop Closing feature of OV²SLAM, you can also comment the lines related to obindex2 and ibow-lcd.

**(Optional)** Install OpenGV:

```
    cd your_path/
    git clone https://github.com/laurentkneip/opengv
    cd opengv
    mkdir build
    cd build/
    cmake ..
    sudo make -j4 install
```


### 2.2 Build OV²SLAM

Build OV²SLAM package with your favorite catkin tool:

```
    cd ~/catkin_ws/src/ov2slam
    catkin build --this
    source ~/catkin_ws/devel/setup.bash
```

OR

```
    cd ~/catkin_ws/
    catkin_make --pkg ov2slam
    source ~/catkin_ws/devel/setup.bash
```

## 3. Usage

Run OV²SLAM using:

```
    rosrun ov2slam ov2slam_node parameter_file.yaml
```

Visualize OV²SLAM outputs in Rviz by loading the provided configuration file: ov2slam_visualization.rviz. 

## 4. Miscellaneous

### Supported Cameras Model

Both the Pinhole Rad-tan and Fisheye camera's models are supported.  The models are OpenCV-based.
If you use [Kalibr](https://github.com/ethz-asl/kalibr) for camera calibration, the equivalencies are: 

- OpenCV "Pinhole" -> Kalibr "Pinhole Radtan" 
- OpenCV "Fisheye" -> Kalibr "Pinhole Equidistant"

### Extrinsic Calibration

The stereo extrinsic parameters in the parameter files are expected to represent the transformation from the camera frame to the body frame (**T_body_cam \ X_body = T_body_cam * X_cam**).
Therefore, if **T_body_camleft** is set as the Identity transformation, for the right camera we have: **T_body_camright** = **T_camleft_camright**.
In Kalibr, the inverse transformation is provided (i.e. **T_cam_body**).  Yet, Kalibr also provide the extrinsic transformation of each camera w.r.t. to the previous one with the field **T_cn_cnm1**.  This transformation can be directly used in OV²SLAM by setting **T_body_camleft** = **T_cn_cnm1** and **T_body_camright** = **I_4x4**.

### Parameters File Description

Three directories are proposed within the parameter_files folder: _*accurate*_, _*average*_ and _*fast*_.  They all store the parameter files to be used with KITTI, EuRoC and TartanAir.

* The _*accurate*_ folder provides the parameters as used in the paper for the full method (i.e. OV²SLAM w. LC).  

* The _*fast*_ folder provides the parameters as used in the paper for the Fast version of OV²SLAM.

* The _*average*_ folder is provided for convenience as an in-between mode.

**TODO**: add description of the parameters file

### Note on "-march=native"

If you experience issues when running OV²SLAM (segfault exceptions, ...), it might be related to the "-march=native" flag.
By default, OpenGV and OV²SLAM come with this flag enabled but Ceres does not.  Making sure that all of them are built with or 
without this flag might solve your problem.