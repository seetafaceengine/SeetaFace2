#!/bin/bash 
#下载工具  

set -ev

SOURCE_DIR="`pwd`"
echo $SOURCE_DIR
TOOLS_DIR=${SOURCE_DIR}/Tools
echo ${TOOLS_DIR}

if [ "$BUILD_TARGERT" = "android" ]; then
    export ANDROID_SDK_ROOT=${TOOLS_DIR}/android-sdk
    export ANDROID_NDK_ROOT=${TOOLS_DIR}/android-ndk
    export JAVA_HOME="/C/Program Files (x86)/Java/jdk1.8.0"
    export PATH=${TOOLS_DIR}/apache-ant/bin:$JAVA_HOME:$PATH
else
    exit 0
fi

if [ ! -d "${TOOLS_DIR}" ]; then
    mkdir ${TOOLS_DIR}
fi

cd ${TOOLS_DIR}

#下载ANT 
wget -c -nv http://apache.fayea.com//ant/binaries/apache-ant-1.10.1-bin.tar.gz
tar xzf apache-ant-1.10.1-bin.tar.gz
rm -f apache-ant-1.10.1-bin.tar.gz
mv apache-ant-1.10.1 apache-ant

#Download android sdk  
if [ ! -d "${TOOLS_DIR}/android-sdk" ]; then
    wget -c -nv https://dl.google.com/android/android-sdk_r24.4.1-windows.zip
    unzip -q android-sdk_r24.4.1-windows.zip
    mv android-sdk-windows android-sdk
    rm android-sdk_r24.4.1-windows.zip
    (sleep 5 ; while true ; do sleep 1 ; printf 'y\r\n' ; done ) \
    | android-sdk/tools/android.bat update sdk -u -t tool,android-18,android-24,extra,platform,platform-tools,build-tools-24.0.1
fi

#下载android ndk  
if [ ! -d "${TOOLS_DIR}/android-ndk" ]; then
    wget -c -nv http://dl.google.com/android/ndk/android-ndk-r10e-windows-x86.exe
    ./android-ndk-r10e-windows-x86.exe > /dev/null
    mv android-ndk-r10e android-ndk
    rm android-ndk-r10e-windows-x86.exe
fi

cd ${SOURCE_DIR}
