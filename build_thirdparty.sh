#!/bin/bash

echo ""
echo "Building Obindex2 lib!"
echo ""

cd Thirdparty/obindex2

mkdir build
cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ../../..

echo ""
echo "Building iBoW-LCD lib!"
echo ""

cd Thirdparty/ibow_lcd

mkdir build
cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ../../..

echo ""
echo "Building Sophus lib!"
echo ""

cd Thirdparty/Sophus

mkdir build
mkdir install
cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="../install/"
make -j4 install
cd ../../..

echo ""
echo "Building Ceres lib!"
echo ""

cd Thirdparty/ceres-solver
mkdir build
mkdir install
cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14 -DCMAKE_CXX_FLAGS="-march=native" -DCMAKE_INSTALL_PREFIX="../install/" -DBUILD_EXAMPLES=OFF
make -j4 install
cd ../../..