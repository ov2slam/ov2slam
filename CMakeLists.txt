cmake_minimum_required(VERSION 2.8.3)
project(ov2slam)

if(POLICY CMP0022)
  cmake_policy(SET CMP0022 NEW)
endif()

#######
# NOTE!
#######
# Set to OFF if you did not compile OpenCV w. opencv_contrib
set(WITH_OPENCV_CONTRIB ON)

if( ${CMAKE_VERSION} VERSION_LESS "3.8.2" )
    set(CMAKE_CXX_STANDARD 14)
else()
    set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_BUILD_TYPE "Release")

#######
# NOTE!
#######
# You might need to remove the -march=native if you experience
# exceptions related to ceres or opengv.
# opengv comes natively with the march=native option while
# ceres requires to specify it at build time.
# If you experience exceptions you can either re-compile
# opengv and ov2slam without -march=native or re-compile
# ceres with it
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -march=native -g")


if(WITH_OPENCV_CONTRIB)
    message("\nGoing to use OpenCV contrib! (WITH_OPENCV_CONTRIB is set to : ${WITH_OPENCV_CONTRIB}.
Set it to OFF is OpenCV was not compiled with the opencv_contrib modules)\n")
    add_definitions(-DOPENCV_CONTRIB)
endif()


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
    COMPONENTS
        roscpp
        std_msgs
        sensor_msgs
        cv_bridge
        image_transport
        tf
        pcl_ros
)

# iBOW-LCD
#
# First check if the related thridparty libs have been
# built. Only set the use of iBoW-LCD to true if built.
set(WITH_IBOW_LCD OFF)

if( EXISTS "${PROJECT_SOURCE_DIR}/Thirdparty/ibow_lcd/build/liblcdetector.so" )
  add_definitions(-DIBOW_LCD)
  set(WITH_IBOW_LCD ON)
  message(STATUS "iBoW-LCD found!  Going to use Loop Closer!")
else ( )
  message(STATUS "iBoW-LCD NOT found!  Loop Closer will not be enabled!")
endif ( )

## System dependencies are found with CMake's conventions

# OpenCV
find_package(OpenCV REQUIRED)

if(OpenCV_VERSION_MAJOR LESS 3)
  message( FATAL_ERROR "OpenCV 3 or 4 is required!  Current version : ${OpenCV_VERSION}" )
endif()

if(OpenCV_VERSION VERSION_LESS "3.3.0")
  message("ENABLE Internal ParallelLoopBodyLambdaWrapper Class definition (OpenCV < 3.3.0)")
  add_definitions(-DOPENCV_LAMBDA_MISSING)
endif()

#Eigen3
find_package(Eigen3 3.3.0 REQUIRED NO_MODULE)

if (Eigen3_FOUND)
  message(STATUS "Found Eigen3 lib : ${Eigen3_VERSION} - ${EIGEN3_INCLUDE_DIRS}")
endif (Eigen3_FOUND)

# add_definitions(-DEIGEN_DONT_VECTORIZE -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)

#OpenGV
find_package(opengv QUIET)
# Bug include CMake 1.17
get_target_property(opengv_INCLUDE_DIR opengv INTERFACE_INCLUDE_DIRECTORIES)

if (opengv_FOUND)
  message(STATUS "Found Opengv lib : ${opengv_INCLUDE_DIR} - ${opengv_DIR}")
  add_definitions(-DUSE_OPENGV)
endif (opengv_FOUND)

# Ceres
find_package(Ceres NO_DEFAULT_PATH PATHS ${PROJECT_SOURCE_DIR}/Thirdparty/ceres-solver/install/)
find_package(Ceres REQUIRED)

if (Ceres_FOUND)
  message(STATUS "Found Ceres lib : ${CERES_VERSION} - ${Ceres_DIR}")
endif (Ceres_FOUND)

#Sophus
# First search for a custom build of Sophus within this project
find_package(Sophus NO_DEFAULT_PATH PATHS ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus/build/)
find_package(Sophus REQUIRED)

if (Sophus_FOUND)
  message(STATUS "Found Sophus lib : ${Sophus_VERSION} - ${Sophus_DIR}")
endif (Sophus_FOUND)

catkin_package()

###########
## Build ##
###########

## add backward
add_subdirectory(Thirdparty/backward-cpp)

## Declare a C++ library
add_library(${PROJECT_NAME}
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
    ${PROJECT_SOURCE_DIR}/include
    ${catkin_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include/ceres_parametrization
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
      ${catkin_LIBRARIES}
      Eigen3::Eigen
      Sophus::Sophus
    PRIVATE
      ${OpenCV_LIBS}
      Ceres::ceres
)


# Add optional dependencies if they are found
if (WITH_IBOW_LCD)
  target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
      ${PROJECT_SOURCE_DIR}/Thirdparty/obindex2/lib/include/
      ${PROJECT_SOURCE_DIR}/Thirdparty/ibow_lcd/include/
  )

  target_link_libraries(
    ${PROJECT_NAME}
      PRIVATE
        ${PROJECT_SOURCE_DIR}/Thirdparty/ibow_lcd/build/liblcdetector.so
  )
endif (WITH_IBOW_LCD)

if (opengv_FOUND)
  target_link_libraries(
    ${PROJECT_NAME}
      PRIVATE
        opengv
  )
endif (opengv_FOUND)

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(${PROJECT_NAME}_node src/ov2slam_node.cpp ${BACKWARD_ENABLE})
add_backward(${PROJECT_NAME}_node)

## Specify libraries to link a library or executable target against
target_link_libraries(
    ${PROJECT_NAME}_node
    PRIVATE
      ${PROJECT_NAME}
)
