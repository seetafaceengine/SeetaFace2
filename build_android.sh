if [ -n "$1" ]; then
    ANDROID_NDK=$1
fi
if [ -z "${ANDROID_NDK}" ]; then
    echo "$0 ANDROID_NDK"
    exit -1
fi

if [ ! -d build ]; then
    mkdir -p build
fi
cd build

cmake .. -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a with NEON" -DANDROID_PLATFORM=android-18 -DBUILD_EXAMPLE=OFF # 如果有OpenCV，则设置为ON

cmake --build . --config Release -- -j`cat /proc/cpuinfo |grep 'cpu cores' |wc -l`

cmake --build . --config Release --target install -- -j`cat /proc/cpuinfo |grep 'cpu cores' |wc -l`

cd ..
