
# **SeetaFace2**

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

[中文](./README.md) [English](./README_en.md)

## 1. 简介
`SeetaFace2` 人脸识别引擎包括了搭建一套全自动人脸识别系统所需的三个核心模块，即：人脸检测模块 `FaceDetector`、面部关键点定位模块 `FaceLandmarker` 以及人脸特征提取与比对模块 `FaceRecognizer`。还将陆续开源人脸跟踪、闭眼检测等辅助模块。

<div align=center>
<img src="./asserts/pipeline.png" width="580" height="230" />
</div>

`SeetaFace2` 采用标准 C++ 开发，全部模块均不依赖任何第三方库，支持 x86 架构（Windows、Linux）和 ARM 架构（Android）。SeetaFace2 支持的上层应用包括但不限于人脸门禁、无感考勤、人脸比对等。

<div align=center>
<img src="./asserts/grid.png" width="630" height="370" />
</div>


SeetaFace2 是面向于人脸识别商业落地的里程碑版本，其中人脸检测模块在 FDDB 上的 100 个误检条件下可达到超过 92% 的召回率，面部关键点定位支持 5 点和 81 点定位，1 比 N 模块支持数千人规模底库的人脸识别应用。


模块 | 方法概述 | 基础技术指标 | 典型平台速度
-----|---------|-------------|------------
**人脸检测** | Cascaded CNN | FDDB 上召回率达到92%（100个误检情况下）。 | 40 最小人脸<br>I7: 70FPS(1920x1080)<br>RK3399: 25FPS(640x480)
**面部关建点定位(81点和5点)** | FEC-CNN | 平均定位误差（根据两眼中心距离归一化）<br>300-W Challenge Set 上达到 0.069。 | I7: 450FPS 和 500FPS<br>RK3399: 110FPS 和 220FPS
**人脸特征提取与比对** | ResNet50 | 识别：通用1：N+1场景下，错误接受率1%时，<br>1000人底库，首选识别率超过98%，<br>5000人底库，首选识别率超过95%。 | I7: 8FPS<br>RK3399: 2.5FPS

与 2016 年开源的 `SeetaFace 1.0` 相比，`SeetaFace2` 在速度和精度两个层面上均有数量级的提升。

<table>
    <tr>
        <th rowspan="2">版本</th>
        <th colspan="2">人脸检测</th>
        <th colspan="2">关键点定位</th>
        <th colspan="2">人脸识别</th>
        <th rowspan="2">第三方依赖</th>
    </tr>
    <tr>
        <td>速度[1]</td>
        <td>单精度[2]</td>
        <td>速度</td>
        <td>功能</td>
        <td>训练数据规模</td>
        <td>应用</td>
    </tr>
    <tr>
        <th>1.0</th>
        <td>16FPS</td>
        <td>85%</td>
        <td>200FPS</td>
        <td>5点</td>
        <td>140万张</td>
        <td>实验室</td>
        <td>无</td>
    </tr>
    <tr>
        <th>2.0</th>
        <td>77FPS</td>
        <td>92%</td>
        <td>500FPS</td>
        <td>5/81点</td>
        <td>3300万张</td>
        <td>商业环境</td>
        <td>无</td>
    </tr>
    <tr>
        <th>备注</th>
        <td colspan="7">
            [1] 640x480输入、检测40x40人脸、I7-6700。<br>
            [2] 人脸检测的精度指100个误捡FDDB数据集的召回率。
        </td>
    </tr>
</table>

知人识面辩万物，开源赋能共发展。`SeetaFace2` 致力于 AI 赋能发展，和行业伙伴一起共同推进人脸识别技术的落地。


## 2. 编译
### 2.1 编译依赖
+ 编译工具
  + For linux
    - GNU Make 工具
    - GCC 或者 Clang 编译器
  + For windows
    - [MSVC](http://msdn.microsoft.com/zh-cn/vstudio) 或者 MinGW. 
  - [CMake](http://www.cmake.org/)
+ 依赖库
  - [可选] [OpneCV](http://opencv.org/) 仅编译例子时需要
+ 依赖架构
  - CPU 支持 SSE2 和 FMA [可选]（x86）或 NENO（ARM）支持

### 2.2 编译参数
  - PLATFORM: [STRING] 编译目标架构，x86/x86_64/amd64 不需要设置，ARM 架构需要设置为对应平台
  - BUILD_DETECOTOR: 是否编译人脸检测模块。ON：打开；OFF：关闭
  - BUILD_LANDMARKER: 是否编译面部关键点定位模块。ON：打开；OFF：关闭
  - BUILD_RECOGNIZER: 是否编译人脸特征提取与比对模块。ON：打开；OFF：关闭
  - BUILD_EXAMPLE: 是否编译例子。ON：打开；OFF：关闭，打开需要预先安装 `OpneCV`
  - CMAKE_INSTALL_PREFIX: 安装前缀
  - SEETA_USE_FMA: 是否启用 `FMA` 指令。默认关闭。只有目标是`x86`架构是起作用

### 2.3 各平台编译
#### 2.3.1 linux平台编译说明
  - 依赖
    + opencv。仅编译例子时需要

        sudo apt-get install libopencv-dev 

  - 编译

        cd SeetaFace2
        mkdir build
        cd build
        cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/install -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLE=OFF # 如果有 OpneCV，则设置为 ON
        cmake --build . --config Release 

    + ARM 架构编译需要制定平台
        ```
        cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/install -DCMAKE_BUILD_TYPE=Release -DPLATFORM=arm
        cmake --build . --config Release 
        ```
  - 安装

        cmake --build .  --config Release --target install

  - 运行例子
    + 把生成库的目录加入到变量 LD_LIBRARY_PATH 中
 
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/lib

    + 拷贝模型文件到程序执行目录的 model 目录下

            cd SeetaFace2
            cd build
            cd bin
            mkdir model
            cp fd_2_00.dat pd_2_00_pts5.dat pd_2_00_pts81.dat .

    + 执行 bin 目录下的程序
      - points81

            cd SeetaFace2
            cd build
            cd bin
            ./points81

      - search

            cd SeetaFace2
            cd build
            cd bin
            ./search

#### 2.3.2 windows平台编译说明
  - 使用 cmake-gui.exe 工具编译。打开 cmake-gui.exe
  - 命令行编译
    + 把 cmake 命令所在目录加入到环境变量 PATH 中
    + 从开始菜单打开 “VS2015开发人员命令提示”，进入命令行

      - 编译

            cd SeetaFace2
            mkdir build
            cd build
            cmake .. -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLE=OFF # 如果有 OpneCV，则设置为 ON
            cmake --build . --config Release 

      - 安装

            cmake --build . --config Release --target install

      - 运行例子
        + 拷贝模型文件到程序执行目录的 model 目录下

                cd SeetaFace2
                cd build
                cd bin
                mkdir model
                cp fd_2_00.dat pd_2_00_pts5.dat pd_2_00_pts81.dat .

        + 执行 bin 目录下的程序
          - points81
          - search

#### 2.3.3 Android平台编译说明
+ 安装 ndk 编译工具
  - 从  https://developer.android.com/ndk/downloads 下载 ndk，并安装到：/home/android-ndk
  - 设置环境变量：

        export ANDROID_NDK=/home/android-ndk

+ 编译
  - 主机是linux

        cd SeetaFace2
        mkdir build
        cd build
        cmake .. -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a with NEON" -DANDROID_PLATFORM=android-18 -DBUILD_EXAMPLE=OFF # 如果有OpenCV，则设置为ON
        cmake --build . --config Release --target install

  - 主机是windows

        cd SeetaFace2
        mkdir build
        cd build
        cmake .. -DCMAKE_INSTALL_PREFIX=install -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake  -DCMAKE_MAKE_PROGRAM=${ANDROID_NDK}/prebuilt/windows-x86_64/bin/make.exe -DANDROID_ABI=arm64-v8a -DANDROID_ARM_NEON=ON -DBUILD_EXAMPLE=OFF # 如果有 OpenCV，则设置为ON
        cmake --build . --config Release --target install

  - 参数说明：https://developer.android.google.cn/ndk/guides/cmake
    + ANDROID_ABI: 可取下列值：
      目标 ABI。如果未指定目标 ABI，则 CMake 默认使用 armeabi-v7a。  
      有效的目标名称为：
      - armeabi：带软件浮点运算并基于 ARMv5TE 的 CPU。
      - armeabi-v7a：带硬件 FPU 指令 (VFPv3_D16) 并基于 ARMv7 的设备。
      - armeabi-v7a with NEON：与 armeabi-v7a 相同，但启用 NEON 浮点指令。这相当于设置 -DANDROID_ABI=armeabi-v7a 和 -DANDROID_ARM_NEON=ON。
      - arm64-v8a：ARMv8 AArch64 指令集。
      - x86：IA-32 指令集。
      - x86_64 - 用于 x86-64 架构的指令集。
    + ANDROID_NDK <path> 主机上安装的 NDK 根目录的绝对路径
    + ANDROID_PLATFORM: 如需平台名称和对应 Android 系统映像的完整列表，请参阅 [Android NDK 原生 API](https://developer.android.google.cn/ndk/guides/stable_apis.html)
    + ANDROID_ARM_MODE
    + ANDROID_ARM_NEON
    + ANDROID_STL:指定 CMake 应使用的 STL。 

### 2.3.4 IOS 平台编译说明
> 以实体机为例

+ 环境准备
  - 需要 MacOS 的 PC。
  - git 下载源代码。

+ 命令行编译
  + 使用 cmake 编译并安装项目，
    ```
    cd SeetaFace2
    mkdir build
    cd build
    chmod +x ../ios/cmake.sh
    ../ios/cmake.sh -DCMAKE_INSTALL_PREFIX=`pwd`/install
    make -j4
    make install
    ```

    执行完毕后，生成好的静态库将安装到`SeetaFace2/build/install`

  + 编译模拟器版本
    修改 cmake 指令参数 `../ios/cmake.sh -DIOS_PLATFORM=SIMULATOR64 -DPLATFORM=x64`

  + 查看 `<root>/ios/cmake.sh` 和 `<root>/ios/iOS.cmake` 获取更多编译选项


## 3. 目录结构


    |-- SeetaFace2<br>
        |-- documents（SDK 接口说明文档）  
        |-- example（C++ 版本 SDK 示例代码）  
        |-- FaceDetector（人脸检测模块）  
        |-- FaceLandmarker（特征点定位模块）  
        |-- FaceRecognizer（人脸特征提取和比对模块）  
        |-- SeetaNet（前向计算框架模块）  

    
## 4. 模型下载
- 人脸检测模块 FaceDetector 模型下载链接：  
MD5     ：E88669E5F1301CA56162DE8AEF1FD5D5  
百度网盘：https://pan.baidu.com/s/1Dt0M6LXeSe4a0Pjyz5ifkg 提取码：fs8r  
Dropbox : https://www.dropbox.com/s/cemt9fl48t5igfh/fd_2_00.dat?dl=0

-  面部特征5点定位模块 FaceLandmarker 模型下载链接：  
MD5     ：877A44AA6F07CB3064AD2828F50F261A  
百度网盘：https://pan.baidu.com/s/1MqofXbmTv8MIxnZTDt3h5A 提取码：7861  
Dropbox : https://www.dropbox.com/s/noy8tien1gmw165/pd_2_00_pts5.dat?dl=0

-  面部特征81点定位模块 FaceLandmarker 模型下载链接：  
MD5     ：F3F812F01121B5A80384AF3C35211BDD  
百度网盘：https://pan.baidu.com/s/1CCfTGaSg_JSY3cN-R1Myaw 提取码：p8mc  
Dropbox : https://www.dropbox.com/s/v41lmclaxpwow1d/pd_2_00_pts81.dat?dl=0

- 人脸特征提取和比对模块 FaceRecognizer 模型下载链接：  
MD5     ：2D637AAD8B1B7AE62154A877EC291C99  
百度网盘：https://pan.baidu.com/s/1y2vh_BHtYftR24V4xwAVWg 提取码：pim2  
Dropbox : https://www.dropbox.com/s/6aslqcokpljha5j/fr_2_10.dat?dl=0

## 5. 示例 
### 5.1 本项目自带示例

`example/search/example.cpp` 示例展示了一套简单且完整的人脸识别的流程，包括：  
  1. 预注册图像中的人脸到人脸识别底库中（example 中默认注册了"1.jpg"中的人脸）；
  2. 打开摄像头，检测摄像头画面中的人脸；3.对检测到人脸进行识别，确定所属人脸的身份。

测试者如果想在底库中成功识别出自己的人脸，需要在example.cpp的底库注册列表部分添加以自己名称命名的图片(名称 + .jpg)，
并把自己名称命名的图片文件拷贝到程序的运行目录下，重新编译 example 并运行程序，测试识别效果即可。

### 5.2 已使用本项目的其它项目

FaceRecognizer: https://github.com/KangLin/FaceRecognizer

## 6. 开发者社区
欢迎开发者加入 SeetaFace 开发者社区，请先加 SeetaFace 小助手微信，经过审核后邀请入群。

![QR](./asserts/QR.png)

## 6.1 代码贡献
欢迎开发者贡献优质代码，所有开发者代码需提交在`develop`分支。

## 7. 商业合作
想要购买 `SeetaFace` 商业版引擎以获得精度更高、速度更快的人脸识别算法或活体验证、表情识别、心率估计、姿态估计、视线追踪等更多人脸分析模块支持，请联系商务邮件 bd@seetatech.com。

## 8. 开源协议

`SeetaFace2` 依照 [BSD 2-Clause license](LICENSE) 开源.

