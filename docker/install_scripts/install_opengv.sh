#!/bin/sh
set -x 
set -e
BUILD_DIR="$1"
OPENGV_VERSION="$2"


USAGE="Usage: $0 BUILD_DIR OPENGV_VERSION"

if [ -z "$OPENGV_VERSION" ]
then
      echo "Error: \$OPENGV_VERSION is not defined. "
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

(cd "$BUILD_DIR/"; git clone https://github.com/laurentkneip/opengv)
(cd "${BUILD_DIR}/opengv"; git checkout -b "commit_to_build" "$OPENGV_VERSION")
mkdir "${BUILD_DIR}/opengv/build"
cd "${BUILD_DIR}/opengv/build"
cmake ..
make -j`nproc` && make install