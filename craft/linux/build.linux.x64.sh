#!/usr/bin/env bash

BUILD_TYPE="Release"
PLATFORM_TARGET="x64"
PLATFORM="x64"
OPENCV_CMAKE_PREFIX_PATH="/usr"


HOME=$(cd `dirname $0`; pwd)

echo "begin compile SeetaNet..."
WORKPATH="$HOME/../../SeetaNet/sources/build"
mkdir $WORKPATH
cd $WORKPATH
rm -rf *


cmake ".." \
-DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
-DCONFIGURATION="$BUILD_TYPE" \
-DPLATFORM="$PLATFORM" \
-DCMAKE_PREFIX_PATH="$CUDA_CMAKE_PREFIX_PATH"

make -j16

cp -rf "include" $HOME/.
if [ "$PLATFORM" = "x64" ]; then
cp -rf "lib64" $HOME/.
else
cp -rf "lib32" $HOME/.
fi 


echo "begin compile FaceDetector..."
WORKPATH="$HOME/../../FaceDetector/build"
mkdir $WORKPATH
cd $WORKPATH
rm -rf *


cmake ".." \
-DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
-DCONFIGURATION="$BUILD_TYPE" \
-DPLATFORM="$PLATFORM" 

make -j16


cp -rf "include" $HOME/.
if [ "$PLATFORM" = "x64" ]; then
cp -rf "lib64" $HOME/.
else
cp -rf "lib32" $HOME/.
fi 


echo "begin compile FaceLandmarker..."
WORKPATH="$HOME/../../FaceLandmarker/build"
mkdir $WORKPATH
cd $WORKPATH
rm -rf *


cmake ".." \
-DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
-DCONFIGURATION="$BUILD_TYPE" \
-DPLATFORM="$PLATFORM" 

make -j16


cp -rf "include" $HOME/.
if [ "$PLATFORM" = "x64" ]; then
cp -rf "lib64" $HOME/.
else
cp -rf "lib32" $HOME/.
fi 



echo "begin compile FaceRecognizer..."
WORKPATH="$HOME/../../FaceRecognizer/build"
mkdir $WORKPATH
cd $WORKPATH
rm -rf *


cmake ".." \
-DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
-DCONFIGURATION="$BUILD_TYPE" \
-DPLATFORM="$PLATFORM" 

make -j16


cp -rf "include" $HOME/.
if [ "$PLATFORM" = "x64" ]; then
cp -rf "lib64" $HOME/.
else
cp -rf "lib32" $HOME/.
fi 


cd $HOME
