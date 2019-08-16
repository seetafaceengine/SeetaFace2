@echo off

::set variables
set BUILD_TYPE="Release"
set PLATFORM="x64"
set PLATFORM_TARGET="x64"
set OPENCV_CMAKE_PREFIX_PATH="C:\thirdparty\opencv\build"



::------------------------------------
set HOME=%~dp0


echo "begin compile SeetaNet..."
set WORKPATH="%HOME%..\..\SeetaNet\sources\build"

rd /s/q %WORKPATH%
md %WORKPATH%
cd %WORKPATH%

cmake .. ^
      -G"NMake Makefiles" ^
      -DCMAKE_BUILD_TYPE="%BUILD_TYPE%" ^
      -DCONFIGURATION="%BUILD_TYPE%" ^
      -DPLATFORM="%PLATFORM_TARGET%" ^
      -DCMAKE_PREFIX_PATH="%CUDA_CMAKE_PREFIX_PATH%"


nmake

xcopy /s/k/y "include" %HOME%include\
if %PLATFORM_TARGET% == "x64" (
    
    xcopy /k/y "lib\x64\*.dll" %HOME%lib\x64\
    xcopy /k/y "lib\x64\*.lib" %HOME%lib\x64\
) else (
    xcopy /k/y "lib\x86\*.dll" %HOME%lib\x86\
    xcopy /k/y "lib\x86\*.lib" %HOME%lib\x86\
)






echo "begin compile FaceDetector..."
set WORKPATH="%HOME%..\..\FaceDetector\build"

rd /s/q %WORKPATH%
md %WORKPATH%
cd %WORKPATH%

cmake .. ^
      -G"NMake Makefiles" ^
      -DCMAKE_BUILD_TYPE="%BUILD_TYPE%" ^
      -DCONFIGURATION="%BUILD_TYPE%" ^
      -DPLATFORM="%PLATFORM_TARGET%"

nmake

xcopy /s/k/y "include" %HOME%include\
if %PLATFORM_TARGET% == "x64" (
    
    xcopy /k/y "lib\x64\*.dll" %HOME%lib\x64\
    xcopy /k/y "lib\x64\*.lib" %HOME%lib\x64\
) else (
    xcopy /k/y "lib\x86\*.dll" %HOME%lib\x86\
    xcopy /k/y "lib\x86\*.lib" %HOME%lib\x86\
)






echo "begin compile FaceLandmarker..."
set WORKPATH="%HOME%..\..\FaceLandmarker\build"

rd /s/q %WORKPATH%
md %WORKPATH%
cd %WORKPATH%


cmake .. ^
      -G"NMake Makefiles" ^
      -DCMAKE_BUILD_TYPE="%BUILD_TYPE%" ^
      -DCONFIGURATION="%BUILD_TYPE%" ^
      -DPLATFORM="%PLATFORM_TARGET%"

nmake

xcopy /s/k/y "include" %HOME%include\
if %PLATFORM_TARGET% == "x64" (
    
    xcopy /k/y "lib\x64\*.dll" %HOME%lib\x64\
    xcopy /k/y "lib\x64\*.lib" %HOME%lib\x64\
) else (
    xcopy /k/y "lib\x86\*.dll" %HOME%lib\x86\
    xcopy /k/y "lib\x86\*.lib" %HOME%lib\x86\
)






echo "begin compile FaceRecognizer..."
set WORKPATH="%HOME%..\..\FaceRecognizer\build"

rd /s/q %WORKPATH%
md %WORKPATH%
cd %WORKPATH%

cmake .. ^
      -G"NMake Makefiles" ^
      -DCMAKE_BUILD_TYPE="%BUILD_TYPE%" ^
      -DCONFIGURATION="%BUILD_TYPE%" ^
      -DPLATFORM="%PLATFORM_TARGET%"

nmake

xcopy /s/k/y "include" %HOME%include\
if %PLATFORM_TARGET% == "x64" (
    
    xcopy /k/y "lib\x64\*.dll" %HOME%lib\x64\
    xcopy /k/y "lib\x64\*.lib" %HOME%lib\x64\
) else (
    xcopy /k/y "lib\x86\*.dll" %HOME%lib\x86\
    xcopy /k/y "lib\x86\*.lib" %HOME%lib\x86\
)


cd %HOME%
exit /b
