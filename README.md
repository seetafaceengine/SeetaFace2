# **SeetaFace2**

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

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
- GNU Make 工具<br>
- GCC 或者 Clang 编译器<br>
- CMake<br>

### 2.2 linux和windows平台编译说明
1. 编译参数
  - BUILD_DETECOTOR: 是否编译人脸检测模块。ON：打开；OFF：关闭
  - BUILD_LANDMARKER: 是否编译面部关键点定位模块。ON：打开；OFF：关闭
  - BUILD_RECOGNIZER: 是否编译人脸特征提取与比对模块。ON：打开；OFF：关闭
  - BUILD_EXAMPLE: 是否编译例子。ON：打开；OFF：关闭
  - CMAKE_INSTALL_PREFIX: 安装前缀

2. linux
  - 信赖
    + opencv。仅编译例子时需要

        sudo apt-get install libopencv-dev 

  - 编译

        cd SeetaFace2
        mkdir build
        cd build
        cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/install
        cmake --build .

  - 安装

        cmake --build . --target install

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
      - point81
      - search

3. windows
  - 信赖
    + opencv。仅编译例子时需要
  - 使用 cmake-gui.exe 。打开 cmake-gui.exe
  - 命令行编译
    + 把 cmake 命令所在目录加入到环境变量 PATH 中
    + 从开始菜单打开 “VS2015开发人员命令提示”，进入命令行

      - 编译

        cd SeetaFace2
        mkdir build
        cd build
        cmake .. -DCMAKE_INSTALL_PREFIX=install
        cmake --build .

      - 安装

        cmake --build . --target install

      - 运行例子
        + 拷贝模型文件到程序执行目录的 model 目录下

            cd SeetaFace2
            cd build
            cd bin
            mkdir model
            cp fd_2_00.dat pd_2_00_pts5.dat pd_2_00_pts81.dat .

        + 执行 bin 目录下的程序
          - point81
          - search

### 2.3 Android平台编译说明
Android 版本的编译方法： 
1. 安装 ndk 编译工具；
2. 环境变量中导出 ndk-build 工具；
2. `cd` 到各模块的 `jni` 目录下（如SeetaNet 的 Android 编译脚本位置为`SeetaNet/sources/jni`， FaceDetector 的 Android 编译脚本位置为`FaceDetector/FaceDetector/jni`），执行 `ndk-build -j8` 命令进行编译。<br>

编译依赖说明：人脸检测模块 `FaceDetector` ， 面部关键点定位模块 `FaceLandmarker` 以及人脸特征提取与比对模块 `FaceRecognizer` 均依赖前向计算框架 `SeetaNet` 模块，因此需优先编译前向计算框架 `SeetaNet` 模块。

## 3. 目录结构
|-- SeetaFace2<br>
&emsp;&emsp;|-- craft（linux 和 windows 平台的编译脚本）<br>
&emsp;&emsp;|-- documents（SDK 接口说明文档）<br>
&emsp;&emsp;|-- example（C++ 版本 SDK 示例代码）<br>
&emsp;&emsp;|-- FaceDetector（人脸检测模块）<br>
&emsp;&emsp;|-- FaceLandmarker（特征点定位模块）<br>
&emsp;&emsp;|-- FaceRecognizer（人脸特征提取和比对模块）<br>
&emsp;&emsp;|-- SeetaNet（前向计算框架模块）<br>

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

## 5. example 说明
`example/search/example.cpp` 示例展示了一套简单且完整的人脸识别的流程，包括：1. 预注册图像中的人脸到人脸识别底库中（example 中默认注册了"1.jpg"中的人脸）；2. 打开摄像头，检测摄像头画面中的人脸；3.对检测到人脸进行识别，确定所属人脸的身份。

测试者如果想在底库中成功识别出自己的人脸，需要在example.cpp的底库注册列表部分添加以自己名称命名的图片(名称 + .jpg)，并把自己名称命名的图片文件拷贝到程序的运行目录下，重新编译 example 并运行程序，测试识别效果即可。

## 6. 开发者社区
欢迎开发者加入 SeetaFace 开发者社区，请先加 SeetaFace 小助手微信，经过审核后邀请入群。

![QR](./asserts/QR.png)

## 7. 商业合作
想要购买 `SeetaFace` 商业版引擎以获得精度更高、速度更快的人脸识别算法或活体验证、表情识别、心率估计、姿态估计、视线追踪等更多人脸分析模块支持，请联系商务邮件 bd@seetatech.com。

## 8. 开源协议

`SeetaFace2` 依照 [BSD 2-Clause license](LICENSE) 开源.
