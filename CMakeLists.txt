#
# Author: Kang Lin<kl222@126.com>
# Date  : 2019-08-20
#

cmake_minimum_required(VERSION 2.8)

project(Seeta)

# Create uninstall files
CONFIGURE_FILE(
    "${CMAKE_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)
# Create unistall target
ADD_CUSTOM_TARGET(uninstall
    "${CMAKE_COMMAND}" -P "${CMAKE_BINARY_DIR}/cmake_uninstall.cmake")

# Set version
# Find Git Version Patch
IF(EXISTS "${CMAKE_SOURCE_DIR}/.git")
    if(NOT GIT)
        SET(GIT $ENV{GIT})
    endif()
    if(NOT GIT)
        FIND_PROGRAM(GIT NAMES git git.exe git.cmd)
    endif()
    IF(GIT)
        EXECUTE_PROCESS(
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMAND ${GIT} describe --tags
            OUTPUT_VARIABLE GIT_VERSION  OUTPUT_STRIP_TRAILING_WHITESPACE
        )
	    if(NOT GIT_VERSION)
            EXECUTE_PROCESS(
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                COMMAND ${GIT} rev-parse --short HEAD
                OUTPUT_VARIABLE GIT_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        endif()
        SET(BUILD_VERSION ${GIT_VERSION})
    ENDIF()
ENDIF()
if(NOT BUILD_VERSION)
    SET(BUILD_VERSION "v2.5.5")
endif()
message("BUILD_VERSION:${BUILD_VERSION}")

IF(MSVC)
    # This option is to enable the /MP switch for Visual Studio 2005 and above compilers
    OPTION(WIN32_USE_MP "Set to ON to build with the /MP option (Visual Studio 2005 and above)." ON)
    MARK_AS_ADVANCED(WIN32_USE_MP)
    IF(WIN32_USE_MP)
        #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
        add_compile_options(/MP)
    ENDIF(WIN32_USE_MP)
    add_compile_options(/utf-8)
ENDIF(MSVC)
IF(MINGW)
    add_compile_options("-std=c++0x")
ENDIF()
SET(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libs")
if (BUILD_SHARED_LIBS)
	add_definitions(-DBUILD_SHARED_LIBS)
    if (CMAKE_COMPILER_IS_GNUCXX AND NOT MINGW)
       # Just setting CMAKE_POSITION_INDEPENDENT_CODE should be enough to set
       # -fPIC for GCC but sometimes it still doesn't get set, so make sure it
       # does.
       add_definitions("-fPIC")
    endif()
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(GenerateExportHeader)
include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckFunctionExists)

#CHECK_INCLUDE_FILE_CXX("string" HAVE_STRING_H)
#check_include_file("math.h" HAVE_MATH_H)

#check_function_exists("fabs" HAVE_FABS)

set(BUILD_PLATFORM "${CMAKE_SYSTEM_NAME}")

# ----------------------------------------------------------------------------
# Detect compiler and target platform architecture
# ----------------------------------------------------------------------------
if(NOT ANDROID)
    if(X86_64 OR CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(PLATFORM x86_64)
    elseif(X86 OR CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(PLATFORM x86)
    endif()
else()
    set(PLATFORM ${CMAKE_ANDROID_ARCH})
endif()

# For debug libs and exes, add "-d" postfix
if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX "-d")
endif()

add_subdirectory(SeetaNet)
option(BUILD_DETECOTOR "Set to ON to build detector" ON)
if(BUILD_DETECOTOR)
    add_subdirectory(FaceDetector)
endif(BUILD_DETECOTOR)
option(BUILD_LANDMARKER "Set to ON to build landmarker" ON)
if(BUILD_LANDMARKER)
    add_subdirectory(FaceLandmarker)
endif(BUILD_LANDMARKER)
option(BUILD_RECOGNIZER "Set to ON to build recognizer" ON)
if(BUILD_RECOGNIZER)
    add_subdirectory(FaceRecognizer)
endif(BUILD_RECOGNIZER)
option(BUILD_EXAMPLE "Set to ON to build example" ON)
if(BUILD_EXAMPLE)
    add_subdirectory(example)
endif(BUILD_EXAMPLE)

# 设置安装前缀
SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/install CACHE STRING "Install path prefix.") 

message("Build detector:${BUILD_DETECOTOR}")
message("Build landmarker:${BUILD_LANDMARKER}")
message("Build recgnizer:${BUILD_RECOGNIZER}")
message("Build example:${BUILD_EXAMPLE}")
message("Build shared library:${BUILD_SHARED_LIBS}")
message("PLATFORM: ${PLATFORM}")
