#!/bin/bash

if [ -n "$1" ]; then
    ANDROID_NDK=$1
fi
if [ -z "${ANDROID_NDK}" ]; then
    echo "$0 ANDROID_NDK"
    exit -1
fi

if [ -z "${ANDROID_STL}" ]; then
   ANDROID_STL=c++_static
fi

if [ ! -d build ]; then
    mkdir -p build
fi
cd build

cmake .. -G"Unix Makefiles" -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=MinSizeRel -DCMAKE_VERBOSE_MAKEFILE=TRUE -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a with NEON" -DANDROID_PLATFORM=android-18 -DANDROID_STL=${ANDROID_STL} -DBUILD_EXAMPLE=OFF # 如果有OpenCV，则设置为ON

cmake --build . --config MinSizeRel -- -j`cat /proc/cpuinfo |grep 'cpu cores' |wc -l`

cmake --build . --config MinSizeRel --target install/strip -- -j`cat /proc/cpuinfo |grep 'cpu cores' |wc -l`

cd ..
