cmake_minimum_required(VERSION 2.8)
set(SeetaFaceDetector_NAME "SeetaFaceDetector" CACHE STRING "The SeetaFaceDetector library name")
set(SeetaFaceDetector_VERSION_MAJOR "2")
set(SeetaFaceDetector_VERSION_MINOR "5")
set(SeetaFaceDetector_VERSION_SINOR "5")
set(SeetaFaceDetector_VERSION "${SeetaFaceDetector_VERSION_MAJOR}${SeetaFaceDetector_VERSION_MINOR}${SeetaFaceDetector_VERSION_SINOR}")
project(${SeetaFaceDetector_NAME}${SeetaFaceDetector_VERSION_MAJOR})

set(SOLUTION_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# option for shared or static
set(TARGET "SHARED" CACHE STRING "SHARED or STATIC" FORCE)


list(APPEND CMAKE_MODULE_PATH ${SOLUTION_DIR}/cmake)
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

# install dir
set(CMAKE_INSTALL_PREFIX ${SOLUTION_DIR}/build/install CACHE STRING "set install prefix" FORCE)
set(SEETA_INSTALL_PATH "" CACHE STRING "seeta install path")
if(SEETA_INSTALL_PATH STREQUAL "")
else()
	set(CMAKE_INSTALL_PREFIX ${SEETA_INSTALL_PATH} CACHE STRING "set install prefix" FORCE)
endif()
message(STATUS "cmake install prefix: ${CMAKE_INSTALL_PREFIX}")

# set base flags
#set(CMAKE_CFLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden -w")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -w -std=c++11")



# option for platform
#set(PLATFORM "x64" CACHE STRING "x86 or x64")
#set(CONFIGURATION "Debug" CACHE STRING "Debug or Release")





set(SEETANET_HOME "${SOLUTION_DIR}/../SeetaNet/sources/build")


# env
include(LOCAL_ENV)


#set flags
if(UNIX)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -W -fPIC -Wsign-compare -Wunused-parameter")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -fPIC -fpermissive -std=c++11 -Wsign-compare -Wunused-parameter")
    if (${CONFIGURATION} STREQUAL "Debug")
        set(CMAKE_C_FLAGS "-O0 -g -ggdb ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-O0 -g -ggdb ${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_C_FLAGS "-O2 ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}")
    endif()
    if (${PLATFORM} STREQUAL "x86")
        set(CMAKE_C_FLAGS "-m32 -msse ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-m32 -msse ${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_C_FLAGS "-m64 -msse ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-m64 -msse ${CMAKE_CXX_FLAGS}")
    endif()
elseif(WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()


add_subdirectory(${SOLUTION_DIR}/FaceDetector)
if(NOT BUILD_ANDROID)
#	add_subdirectory(${SOLUTION_DIR}/example)
endif()
