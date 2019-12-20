#!/bin/bash
set -e

SOURCE_DIR=`pwd`
if [ -n "$1" ]; then
    SOURCE_DIR=$1
fi

cd ${SOURCE_DIR}

if [ "$BUILD_TARGERT" = "android" ]; then
    export ANDROID_SDK_ROOT=${SOURCE_DIR}/Tools/android-sdk
    export ANDROID_NDK_ROOT=${SOURCE_DIR}/Tools/android-ndk
    export ANDROID_SDK=${ANDROID_SDK_ROOT}
    export ANDROID_NDK=${ANDROID_NDK_ROOT}
    if [ -n "$APPVEYOR" ]; then
        export JAVA_HOME="/C/Program Files (x86)/Java/jdk1.8.0"
    fi
    if [ "$TRAVIS" = "true" ]; then
        export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
    fi
    export PATH=${SOURCE_DIR}/Tools/apache-ant/bin:$JAVA_HOME:$PATH
fi

cd ${SOURCE_DIR}

if [ "$BUILD_TARGERT" != "windows_msvc" ]; then
    RABBIT_MAKE_JOB_PARA="-j`cat /proc/cpuinfo |grep 'cpu cores' |wc -l`"  #make 同时工作进程参数
    if [ "$RABBIT_MAKE_JOB_PARA" = "-j1" ];then
        RABBIT_MAKE_JOB_PARA="-j2"
    fi
fi

if [ "$BUILD_TARGERT" = "windows_mingw" \
    -a -n "$APPVEYOR" ]; then
    export PATH=/C/Qt/Tools/mingw${TOOLCHAIN_VERSION}/bin:$PATH
fi
TARGET_OS=`uname -s`
case $TARGET_OS in
    MINGW* | CYGWIN* | MSYS*)
        export PKG_CONFIG=/c/msys64/mingw32/bin/pkg-config.exe
        ;;
    Linux* | Unix*)
    ;;
    *)
    ;;
esac

export PATH=${QT_ROOT}/bin:$PATH
echo "PATH:$PATH"
echo "PKG_CONFIG:$PKG_CONFIG"
cd ${SOURCE_DIR}

mkdir -p build_${BUILD_TARGERT}
cd build_${BUILD_TARGERT}

case ${BUILD_TARGERT} in
    windows_msvc)
        MAKE=nmake
        CONFIG_PARA="${CONFIG_PARA} -DBUILD_EXAMPLE=OFF"
        ;;
    windows_mingw)
        if [ "${RABBIT_BUILD_HOST}"="windows" ]; then
            MAKE="mingw32-make ${RABBIT_MAKE_JOB_PARA}"
            CONFIG_PARA="${CONFIG_PARA} -DBUILD_EXAMPLE=OFF"
        fi
        ;;
    *)
        MAKE="make ${RABBIT_MAKE_JOB_PARA}"
        ;;
esac

if [ -n "${BUILD_SHARED_LIBS}" ]; then
    CONFIG_PARA="${CONFIG_PARA} -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
fi

if [ -n "${ANDROID_ARM_NEON}" ]; then
    CONFIG_PARA="${CONFIG_PARA} -DANDROID_ARM_NEON=${ANDROID_ARM_NEON}"
fi
echo "PWD:`pwd`"
if [ "${BUILD_TARGERT}" = "android" ]; then
    TAR_FILE="SeetaFace_${BUILD_TARGERT}_${BUILD_ARCH}_${ANDROID_API}.tar.gz"
    cmake -G"${GENERATORS}" ${SOURCE_DIR} ${CONFIG_PARA} \
        -DBUILD_EXAMPLE=OFF \
        -DCMAKE_INSTALL_PREFIX=`pwd`/install \
        -DCMAKE_VERBOSE=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DANDROID_PLATFORM=${ANDROID_API} -DANDROID_ABI="${BUILD_ARCH}" \
        -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake
    
else
    cmake -G"${GENERATORS}" ${SOURCE_DIR} ${CONFIG_PARA} \
        -DCMAKE_INSTALL_PREFIX=`pwd`/install \
        -DCMAKE_VERBOSE=ON \
        -DCMAKE_BUILD_TYPE=Release
fi
cmake --build . --config Release --target install -- ${RABBIT_MAKE_JOB_PARA}

if [ "${BUILD_TARGERT}" = "unix" -a "ON" = "${BUILD_SHARED_LIBS}" ]; then
    # configure C compiler
    export compiler=$(which gcc)
    # get version code
    MAJOR=$(echo __GNUC__ | $compiler -E -xc - | tail -n 1)
    MINOR=$(echo __GNUC_MINOR__ | $compiler -E -xc - | tail -n 1)
    PATCHLEVEL=$(echo __GNUC_PATCHLEVEL__ | $compiler -E -xc - | tail -n 1)
    TAR_FILE="SeetaFace_${BUILD_TARGERT}_gcc${MAJOR}.${MINOR}.${PATCHLEVEL}.tar.gz"
fi
if [ -n "${TAR_FILE}" -a "$TRAVIS_TAG" != "" ]; then
    TAR_FILE=`echo "${TAR_FILE}" | sed 's/[ ][ ]*/_/g'`
    cd `pwd`/install
    tar czf "${TAR_FILE}" *
    wget -c https://github.com/probonopd/uploadtool/raw/master/upload.sh
    chmod u+x upload.sh
    ./upload.sh "${TAR_FILE}"
fi
