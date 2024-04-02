cmake_minimum_required(VERSION 3.22)
project(ov2slam)

if(POLICY CMP0022)
  cmake_policy(SET CMP0022 NEW)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

########
# OpenCV
########
find_package(OpenCV REQUIRED)

#######
# NOTE!
#######
# Set to OFF if you did not compile OpenCV w. opencv_contrib
set(WITH_OPENCV_CONTRIB ON)

if (OpenCV_FOUND)
  message(STATUS "Found OpenCV lib : ${OpenCV_VERSION} - ${OpenCV_INCLUDE_DIRS}")
  if(WITH_OPENCV_CONTRIB)
    message(STATUS "Going to use OpenCV contrib!")
    add_definitions(-DOPENCV_CONTRIB)
  endif()
endif (OpenCV_FOUND)

########
# Eigen3
########
find_package(Eigen3 REQUIRED)

if (Eigen3_FOUND)
  message(STATUS "Found Eigen3 lib : ${Eigen3_VERSION} - ${EIGEN3_INCLUDE_DIRS}")
endif (Eigen3_FOUND)

########
# OpenGV
########
find_package(opengv)

if (opengv_FOUND)
  message(STATUS "Found Opengv lib : ${opengv_INCLUDE_DIR} - ${opengv_DIR}")
  add_definitions(-DUSE_OPENGV)
endif (opengv_FOUND)

#######
# Ceres
#######
find_package(Ceres REQUIRED)

if (Ceres_FOUND)
  message(STATUS "Found Ceres lib : ${CERES_VERSION} - ${Ceres_DIR}")
endif (Ceres_FOUND)

########
# Sophus
########
find_package(Sophus REQUIRED)

if (Sophus_FOUND)
  message(STATUS "Found Sophus lib : ${Sophus_VERSION} - ${Sophus_DIR}")
endif (Sophus_FOUND)

##########
# iBOW-LCD
##########
find_package(obindex2)
find_package(lcdetector)

set(WITH_IBOW_LCD OFF)

if (lcdetector_FOUND)
  set(WITH_IBOW_LCD ON)
  add_definitions(-DIBOW_LCD)
  message(STATUS "iBoW-LCD found!  Going to use Loop Closer! : ${lcdetector_VERSION} - ${lcdetector_DIR}")
endif (lcdetector_FOUND)

#######################
## Find ROS packages ##
#######################
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(backward_ros REQUIRED)

###########
## Build ##
###########

## Declare a C++ library
add_library(${PROJECT_NAME} SHARED
  src/ov2slam.cpp
  src/camera_visualizer.cpp
  src/visual_front_end.cpp
  src/frame.cpp
  src/slam_params.cpp
  src/camera_calibration.cpp
  src/feature_extractor.cpp
  src/feature_tracker.cpp
  src/map_manager.cpp
  src/map_point.cpp
  src/mapper.cpp
  src/multi_view_geometry.cpp
  src/ceres_parametrization.cpp
  src/optimizer.cpp
  src/estimator.cpp
  src/loop_closer.cpp
)

# Add mandatory dependencies
target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/include/ceres_parametrization/
)

target_link_libraries(${PROJECT_NAME}
  Eigen3::Eigen
  Sophus::Sophus
  Ceres::ceres
  ${OpenCV_LIBRARIES}
)

if (lcdetector_FOUND)
  target_include_directories(
    ${PROJECT_NAME}
      PRIVATE
      ${PROJECT_SOURCE_DIR}/thirdparty/obindex2/lib/include/
      ${PROJECT_SOURCE_DIR}/thirdparty/ibow_lcd/include/   # should be found by cmake
  )
  target_link_libraries(
    ${PROJECT_NAME}
      lcdetector::lcdetector
  )
endif (lcdetector_FOUND)

if (opengv_FOUND)
  target_link_libraries(
    ${PROJECT_NAME}
      opengv
  )
endif (opengv_FOUND)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  tf2
  tf2_ros
  visualization_msgs
  pcl_conversions
  PCL
  std_msgs
  cv_bridge
  backward_ros
)

install(DIRECTORY include/ DESTINATION include/)
install(TARGETS ${PROJECT_NAME} DESTINATION lib/)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(${PROJECT_NAME}_node src/ov2slam_node.cpp)

ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  image_transport
  std_msgs
  sensor_msgs
  cv_bridge
)

## Add mandatory dependencies
target_include_directories(${PROJECT_NAME}_node
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/include/ceres_parametrization/
)

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}
)

if (lcdetector_FOUND)
  target_include_directories(
    ${PROJECT_NAME}_node
      PRIVATE
      ${PROJECT_SOURCE_DIR}/thirdparty/obindex2/lib/include/
      ${PROJECT_SOURCE_DIR}/thirdparty/ibow_lcd/include/   # should be found by cmake
  )
  target_link_libraries(
    ${PROJECT_NAME}_node
      obindex2::obindex2
      lcdetector::lcdetector
  )
endif (lcdetector_FOUND)

install(TARGETS
  ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
)

ament_package()
