#
# Author: Kang Lin<kl222@126.com>
# Date  : 2019-08-20
#

cmake_minimum_required(VERSION 2.8)

project(SeetaFace)

# Create uninstall files
CONFIGURE_FILE(
    "${CMAKE_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
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
message(STATUS "== BUILD_VERSION: ${BUILD_VERSION}")

if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif(NOT DEFINED CMAKE_BUILD_TYPE)
string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type)

if(WIN32 OR UNIX)
    # default used SSE2
    if(ANDROID OR MINGW)
	    option(SEETA_USE_SSE2 "Set to ON to use SSE2" OFF)
    else()
	    option(SEETA_USE_SSE2 "Set to ON to use SSE2" ON)
    endif()
    if (SEETA_USE_SSE2) 
        add_definitions(-DSEETA_USE_SSE2)
    endif ()
    # default not used FMA
    option(SEETA_USE_FMA "Set to ON to use FMA" OFF)
    if (SEETA_USE_FMA) 
        add_definitions(-DSEETA_USE_FMA)
    endif ()
endif()

IF(MSVC)
    # This option is to enable the /MP switch for Visual Studio 2005 and above compilers
    OPTION(WIN32_USE_MP "Set to ON to build with the /MP option (Visual Studio 2005 and above)." ON)
    MARK_AS_ADVANCED(WIN32_USE_MP)
    IF(WIN32_USE_MP)
        add_compile_options(/MP)
    ENDIF(WIN32_USE_MP)
    add_compile_options(/utf-8)
ENDIF(MSVC)
IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

    add_compile_options(-std=c++0x -Wno-sign-compare -Wno-unused-parameter)
    
    if("debug" STREQUAL build_type)
        add_compile_options(-g -ggdb)
    else()
        add_compile_options(-O3)
    endif()
    
#    if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86")
#        add_compile_options(-m32)
#    elseif ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x64")
#        add_compile_options(-m64)
#    elseif ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
#        add_compile_options(-m64)
#    endif()

    if(SEETA_USE_SSE2)
        add_compile_options(-msse2)
    endif()
    if(SEETA_USE_FMA)
        add_compile_options(-mfma)
    endif()

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

# For debug libs and exes, add "_d" postfix
if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX "_d")
endif()

add_subdirectory(SeetaNet)
option(BUILD_DETECOTOR "Set to ON to build detector" ON)
if(BUILD_DETECOTOR)
    add_subdirectory(FaceDetector)
    add_subdirectory(FaceTracker)
endif(BUILD_DETECOTOR)
option(BUILD_LANDMARKER "Set to ON to build landmarker" ON)
if(BUILD_LANDMARKER)
    add_subdirectory(FaceLandmarker)
endif(BUILD_LANDMARKER)
option(BUILD_RECOGNIZER "Set to ON to build recognizer" ON)
if(BUILD_RECOGNIZER)
    add_subdirectory(FaceRecognizer)
endif(BUILD_RECOGNIZER)
option(BUILD_ASSESSOR "Set to ON to build quality assessor" ON)
if(BUILD_ASSESSOR)
    add_subdirectory(QualityAssessor)
endif(BUILD_ASSESSOR)
option(BUILD_EXAMPLE "Set to ON to build example" OFF)
if(BUILD_EXAMPLE)
    add_subdirectory(example)
endif(BUILD_EXAMPLE)

# Install configure file
configure_file(SeetaFace.pc.in ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

# Install cmake configure files
install(
        FILES cmake/${PROJECT_NAME}Config.cmake
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/SeetaFace2"
)

message(STATUS "== Build detector and tracker: ${BUILD_DETECOTOR}")
message(STATUS "== Build landmarker: ${BUILD_LANDMARKER}")
message(STATUS "== Build recgnizer: ${BUILD_RECOGNIZER}")
message(STATUS "== Build quality assessor: ${BUILD_ASSESSOR}")
message(STATUS "== Build example: ${BUILD_EXAMPLE}")
message(STATUS "== Build shared library: ${BUILD_SHARED_LIBS}")
message(STATUS "== CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "== CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
