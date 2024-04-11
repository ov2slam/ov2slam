#!/bin/sh
set -x 
set -e

BUILD_DIR="$1"
OPENCV_VERSION="$2"

echo "running $@" 

USAGE="Usage: $0 BUILD_DIR OPENCV_VERSION"


if [ -z "$OPENCV_VERSION" ]
then
      echo "Error: \$OPENCV_VERSION is not defined. "
      echo $USAGE
      exit 1
fi

if [ -z "$BUILD_DIR" ]
then
      echo "Error: \$BUILD_DIR is not defined. "
      echo $USAGE
      exit 1
fi


mkdir -p "$BUILD_DIR/"
(cd "${BUILD_DIR}"; \
git clone https://github.com/opencv/opencv.git --branch "${OPENCV_VERSION}" --single-branch --depth 1; \
git clone https://github.com/opencv/opencv_contrib.git --branch "${OPENCV_VERSION}" --single-branch --depth 1 )

mkdir "${BUILD_DIR}/opencv/build"

cd "${BUILD_DIR}/opencv/build"
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH="${BUILD_DIR}/opencv_contrib/modules" \
    -D BUILD_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_IPP=OFF \
    -D -DBUILD_TESTS=OFF \
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
        -DENABLE_PRECOMPILED_HEADERS=OFF \
    ..



make -j`nproc`
make install