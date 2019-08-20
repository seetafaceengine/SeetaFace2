#
# Author: Kang Lin<kl222@126.com>
# Date  : 2019-08-20
#

cmake_minimum_required(VERSION 2.8)
project(SeetaNet)

# option for install
#set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/install CACHE STRING "set install prefix" FORCE)

# set flags
if(UNIX)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -W -fPIC -Wsign-compare -Wunused-parameter")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -fPIC -fpermissive -std=c++11 -Wsign-compare -Wunused-parameter")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(CMAKE_C_FLAGS "-O0 -g -ggdb ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-O0 -g -ggdb ${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_C_FLAGS "-O2 ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}")
    endif()
    if ("${PLATFORM}" STREQUAL "x86")
        set(CMAKE_C_FLAGS "-m32  -msse2 -mfma ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-m32 -msse2 -mfma ${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_C_FLAGS "-m64 -msse2 -mfma ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-m64 -msse2 -mfma ${CMAKE_CXX_FLAGS}")
    endif()
elseif(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

# search files
FILE(GLOB_RECURSE INCLUDE_FILES
        ${PROJECT_SOURCE_DIR}/include/*.h
        ${PROJECT_SOURCE_DIR}/include/*.hpp
        )
FILE(GLOB_RECURSE SRC_FILES
        ${PROJECT_SOURCE_DIR}/src/*.c
        ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/src/*.cc
        )
FILE(GLOB_RECURSE SRC_INCLUDE_FILES
        ${PROJECT_SOURCE_DIR}/src/*.h
        ${PROJECT_SOURCE_DIR}/src/*.hpp
        )

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
    
# for compile
add_library(${PROJECT_NAME} ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/include_inner
    ${PROJECT_SOURCE_DIR}/src/include_inner/layers
    ${PROJECT_SOURCE_DIR}/src
    )

set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${BUILD_VERSION})
# Be will to install header files
set_property(
    TARGET
        ${PROJECT_NAME}
    PROPERTY
        PUBLIC_HEADER ${INCLUDE_FILES}
)

# link library
if (UNIX) 
    target_link_libraries(${PROJECT_NAME} -lpthread)   
endif()

INSTALL(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Config
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/seeta
    )
export(TARGETS ${PROJECT_NAME}
       APPEND FILE ${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake
)
# Install cmake configure files
install(
        EXPORT ${PROJECT_NAME}Config
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
)
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${BUILD_VERSION}
    COMPATIBILITY AnyNewerVersion)
install(FILES "${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake")

