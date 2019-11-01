#!/bin/bash 
#下载工具  

set -e
SOURCE_DIR="`pwd`"
echo $SOURCE_DIR
TOOLS_DIR=${SOURCE_DIR}/Tools

function function_install_yasm()
{
    #安装 yasm
    mkdir -p ${SOURCE_DIR}/Tools/src
    cd ${SOURCE_DIR}/Tools/src
    wget -c -nv http://www.tortall.net/projects/yasm/releases/yasm-1.3.0.tar.gz 
    tar xzf yasm-1.3.0.tar.gz
    cd yasm-1.3.0/
    ./configure > /dev/null && sudo make install -j2 > /dev/null
    cd ${SOURCE_DIR}
}

function function_common()
{
    cd ${SOURCE_DIR}/Tools
    #下载最新cmake程序
    if [ "cmake" = "${QMAKE}" ]; then
        if [ ! -d "`pwd`/cmake" ]; then
            wget -nv --no-check-certificate http://www.cmake.org/files/v3.6/cmake-3.6.1-Linux-x86_64.tar.gz
            tar xzf cmake-3.6.1-Linux-x86_64.tar.gz
            mv cmake-3.6.1-Linux-x86_64 cmake
        fi
    fi
}

function function_android()
{
    cd ${SOURCE_DIR}/Tools
    
    # install oracle jdk
    #sudo add-apt-repository ppa:linuxuprising/java -y
    #sudo apt update
    #(sleep 5 ; while true ; do sleep 1 ; printf '\r\n' ; done ) | sudo apt install oracle-java11-installer -qq -y
    
    #sudo apt install oracle-java11-set-default -qq -y

    #下载android ndk  
    if [ ! -d "`pwd`/android-ndk" ]; then
        if [ "$QT_VERSION_DIR" = "5.9" ]; then
            wget -c -nv http://dl.google.com/android/ndk/android-ndk-r10e-linux-x86_64.bin
            chmod u+x android-ndk-r10e-linux-x86_64.bin
            ./android-ndk-r10e-linux-x86_64.bin > /dev/null
            mv android-ndk-r10e android-ndk
            rm android-ndk-r10e-linux-x86_64.bin
        else
            NDK_VERSION=r20
            wget -c -nv https://dl.google.com/android/repository/android-ndk-${NDK_VERSION}-linux-x86_64.zip
            unzip android-ndk-${NDK_VERSION}-linux-x86_64.zip
            mv android-ndk-${NDK_VERSION} android-ndk
            rm android-ndk-${NDK_VERSION}-linux-x86_64.zip
        fi
    fi

    cd ${SOURCE_DIR}/Tools
    
    #Download android sdk  
    if [ ! -d "`pwd`/android-sdk" ]; then
        wget -c -nv https://dl.google.com/android/android-sdk_r24.4.1-linux.tgz
        tar xf android-sdk_r24.4.1-linux.tgz 
        mv android-sdk-linux android-sdk
        rm android-sdk_r24.4.1-linux.tgz 
        (sleep 5 ; while true ; do sleep 1 ; printf 'y\r\n' ; done ) \
        | android-sdk/tools/android update sdk -u #-t tool,android-18,android-24,extra,platform,platform-tools,build-tools-28.0.3
    fi

    sudo apt-get install ant -qq -y
    sudo apt-get install libicu-dev -qq -y
    
    function_common
    cd ${SOURCE_DIR}
}

function function_unix()
{
    #汇编工具yasm
    #function_install_yasm

    sudo apt-get update -y -qq
    sudo apt-get install debhelper fakeroot -y -qq
    sudo apt-get install -y -qq libglu1-mesa-dev \
        libxkbcommon-x11-dev \
        libpulse-mainloop-glib0 \
        libopencv-dev

    function_common

    cd ${SOURCE_DIR}
}

function function_mingw()
{
    #汇编工具yasm
    #function_install_yasm

    cd ${SOURCE_DIR}
    if [ "true" == "$RABBITIM_BUILD_THIRDLIBRARY" ]; then
        export RABBITIM_BUILD_CROSS_HOST=i686-w64-mingw32 #i586-mingw32msvc
    fi

    function_common
    cd ${SOURCE_DIR}
}

case ${BUILD_TARGERT} in
    android)
        function_android
        ;;
    unix)
        function_unix
        ;;
    windows_mingw)
        function_mingw
        ;;
    *)
    echo "There aren't ${BUILD_TARGERT}"
        ;;
esac

cd ${SOURCE_DIR}
