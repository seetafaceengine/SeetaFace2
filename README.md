# **SeetaFaceEngine2**

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

## 1. 简介
SeetaFaceEngine2人脸识别引擎包括了搭建一套全自动人脸识别系统所需的三个核心模块，即：人脸检测模块FaceDetector、面部特征点定位模块FaceLandmarker以及人脸特征提取与比对模块FaceRecognizer。

## 2. 编译
### 2.1 编译依赖
- GNU Make工具<br>
- GCC或者Clang编译器<br>
- CMake<br>

### 2.2 linux和windows平台编译说明
linux和windows上的SDK编译脚本见目录craft，其中craft/linux下为linux版本的编译脚本，craft/windows下为windows版本的编译脚本，默认编译的库为64位Release版本。

linux和windows上的SDK编译方法：
1. 打开终端（windows上为VS2015 x64 Native Tools Command Prompt工具，linux上为bash），cd到编译脚本所在目录；<br>
2. 执行对应平台的编译脚本。<br>

linux上example的编译运行方法：
1. cd 到example/search目录下，执行make指令；
2. 拷贝模型文件到程序指定的目录下；
3. 执行脚本run.sh。

windows上example的编译运行方法：
1. 使用vs2015 打开SeetaExample.sln构建工程，修改opencv3.props属性表中变量OpenCV3Home的值为本机上的OpenCV3的安装目录；
2. 执行vs2015中的编译命令；
3. 拷贝模型文件到程序指定的目录下，运行程序。

### 2.3 Android平台编译说明
Android版本的编译方法： 
1. 安装ndk编译工具；
2. 环境变量中导出ndk-build工具；
2. cd到各模块的jni目录下（如SeetaNet的Android编译脚本位置为SeetaNet/sources/jni， FaceDetector的Android编译脚本位置为FaceDetector/FaceDetector/jni），执行 ndk-build -j8 命令进行编译。<br>

编译依赖说明：人脸检测模块FaceDetector, 面部特征点定位模块FaceLandmarker以及人脸特征提取与比对模块 FaceRecognizer均依赖前向计算框架SeetaNet模块,因此需优先编译前向计算框架SeetaNet模块。

## 3. 目录结构
|-- SeetaFaceEngine2<br>
&emsp;&emsp;|-- craft（linux和windows平台的编译脚本）<br>
&emsp;&emsp;|-- documents（sdk接口说明文档）<br>
&emsp;&emsp;|-- example（C++版本SDK示例代码）<br>
&emsp;&emsp;|-- FaceDetector（人脸检测模块）<br>
&emsp;&emsp;|-- FaceLandmarker（特征点定位模块）<br>
&emsp;&emsp;|-- FaceRecognizer（人脸特征提取和比对模块）<br>
&emsp;&emsp;|-- SeetaNet（前向计算框架模块）<br>

## 4. 模型下载
- 人脸检测模块FaceDetector模型下载链接：https://pan.baidu.com/s/1Dt0M6LXeSe4a0Pjyz5ifkg 提取码：fs8r
-  面部特征5点定位模块FaceLandmarker模型下载链接：https://pan.baidu.com/s/1MqofXbmTv8MIxnZTDt3h5A 提取码：7861 
-  面部特征81点定位模块FaceLandmarker模型下载链接：https://pan.baidu.com/s/1CCfTGaSg_JSY3cN-R1Myaw 提取码：p8mc
- 人脸特征提取和比对模块FaceRecognizer模型下载链接：https://pan.baidu.com/s/1y2vh_BHtYftR24V4xwAVWg 提取码：pim2 

## 5. example说明
example/search/example.cpp 示例展示了一套简单且完整的人脸识别的流程，包括1. 预注册图像中的人脸到人脸识别底库中（example中默认注册了"1.jpg"中的人脸）；2. 打开摄像头，检测摄像头画面中的人脸；3.对检测到人脸进行识别，确定所属人脸的身份。 <br/>
测试者如果想在底库中成功识别出自己的人脸，需要在example.cpp的底库注册列表部分添加以自己名称命名的图片(名称 + .jpg)，并把自己名称命名的图片文件拷贝到程序的运行目录下，重新编译example并运行程序，测试识别效果即可。
