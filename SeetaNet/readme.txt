Linux seetanet 编译说明：

+==============================================================================+
编译依赖的第三方库：
首先下载最新的 OpenBLAS 库和 
确保系统中的 OpenBLAS 不会与即将编译的内容冲突，可以提前将系统存在的
依赖库删除。

+------------------------------------------------------------------------------+
编译 OpenBLAS x64
首先解压进入 OpenBLAS 根目录（OpenBLAS_Home）
执行下面的指令编译安装 64 位 OpenBLAS：

make -j4 BINARY=64 DYNAMIC_ARCH=1 NUM_THREADS=4 FC=gfortran
sudo make PREFIX=/opt/OpenBLAS install

+------------------------------------------------------------------------------+
编译 OpenBLAS x86
首先解压进入 OpenBLAS 根目录（OpenBLAS_Home）
执行下面的指令编译安装 32 位 OpenBLAS：

make -j4 BINARY=32 DYNAMIC_ARCH=1 NUM_THREADS=4 FC=gfortran
sudo make PREFIX=/opt/OpenBLAS32 install


+------------------------------------------------------------------------------+
补充说明：在同一个目录下编译 64 位和 32 位可能存在冲突，请执行 make clean 保证重
新编译。
OpenBLAS 的 make clean 可能删除不干净，可以采用下面的三个指令（在源码跟目录中执
行）来删除所有临时文件。

find . -name "*.o" -exec rm {} \;
find . -name "*.a" -exec rm {} \;
find . -name "*.so" -exec rm {} \;

+==============================================================================+
将编译好的库加入到 seetanet 的编译依赖中：
首先进入 seetanet 的根目录（seetanet_Home）拷贝依赖的头文件：

# 该指令的目标文件夹可能不存在，则提前创建。
cp /opt/OpenBLAS/include/* 3rdparty/include/openblas/x64
cp /opt/OpenBLAS32/include/* 3rdparty/include/openblas/x86

# 软连接依赖的 so 文件：
ln -s /opt/OpenBLAS/lib/libopenblas.so 3rdparty/lib
ln -s /opt/OpenBLAS32/lib/libopenblas.so 3rdparty/lib32

+==============================================================================+
编译 seetanet：
这里列出了如何同时 32 位和 64 位库，如果不需要对应版本，则对应步骤可以省略。

+------------------------------------------------------------------------------+
编译 seetanet x64:

# <seetanet_Home> 是 seetanet 源代码的根目录。
cd <seetanet_Home>
mkdir build
cd build
cmake -DX64_ARCH=ON ..
make -j4
make install

+------------------------------------------------------------------------------+
默认安装路径为 根目录下的 install 文件，这个时候将生成的文件移走，以生成对应的
32 位库。

cd <seetanet_Home>
mv install install_x64
# 删除 build 文件夹开始重新编译
rm -r build


+------------------------------------------------------------------------------+
编译 seetanet x86:

# <seetanet_Home> 是 seetanet 源代码的根目录。
cd <seetanet_Home>
mkdir build
cd build
cmake ..
make -j4
make install

+------------------------------------------------------------------------------+
将生成的 x86 库移走：
cd <seetanet_Home>
mv install install_x86
rm -r build

+==============================================================================+
文件打包：

# <seetanet_Home> 是 seetanet 源代码的根目录。
cd <seetanet_Home>
cp 3rdparty/lib/*.so install_x64/lib
cp 3rdparty/lib32/*.so install_x86/lib

这个时候可以将所有文件放入最终打包的目录：

# <PREFIX> 为安装目录
cp install_x64/include/* <PREFIX>/include
cp install_x64/lib/* <PREFIX>/lib/x64
cp install_x86/lib/* <PREFIX>/lib/x86

+==============================================================================+
编译测试：
编译的时候除了要链接 seetanet 外还需要链接 openblas：
例如：

g++ main.cpp -lseetanet -lopenblas

+==============================================================================+
随源码附带的编译好的二进制文件是基于 Ubuntu 16.04 GCC 5.4 编译的。

中科视拓
2019年7月25日
