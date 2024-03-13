#
# Author: Kang Lin<kl222@126.com>
# Date  : 2019-08-20
#

cmake_minimum_required(VERSION 2.8)

project("tracking")

if(UNIX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
endif()

find_package(OpenCV REQUIRED)

# add library
add_executable(${PROJECT_NAME} example.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/SeetaNet/include
    ${CMAKE_SOURCE_DIR}/FaceTracker/include
    )
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    )

target_link_libraries(${PROJECT_NAME} PRIVATE
    SeetaNet SeetaFaceTracker)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
# Set flags
if(ANDROID)
    if(ANDROID_ARM_NEON)
        target_compile_definitions(${PROJECT_NAME} PRIVATE __ARM_NEON__)
        target_compile_options(${PROJECT_NAME} PRIVATE -mfpu=neon-vfpv4)
    endif()
    
    target_compile_options(${PROJECT_NAME} PRIVATE 
        -frtti -fexceptions -funsafe-math-optimizations -ftree-vectorize -ffast-math)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    VERSION ${BUILD_VERSION}
    INSTALL_RPATH
        "$ORIGIN;$ORIGIN/../lib;$ORIGIN/../lib/${CMAKE_LIBRARY_ARCHITECTURE};lib;../lib;../lib/${CMAKE_LIBRARY_ARCHITECTURE}"
    )
if(UNIX AND NOT ANDROID)
    set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_NAME ${PROJECT_NAME})
endif()
INSTALL(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )

# Find user and system name
# SET(SYSTEM_NAME $ENV{USERDOMAIN} CACHE STRING SystemName)
# SET(USER_NAME $ENV{USERNAME} CACHE STRING UserName)

if (MSVC_IDE)
    # Configure the template file
    SET(_CMAKE_LocalDebuggerEnvironment "PATH=${CMAKE_BINARY_DIR}/bin/$(Configuration);${CMAKE_BINARY_DIR}/lib/$(Configuration);${OpenCV_LIB_PATH}/../bin;%PATH%")
    SET(USER_FILE ${PROJECT_NAME}.vcxproj.user)
    SET(OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/${USER_FILE})
    CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/UserTemplate.xml ${USER_FILE} @ONLY)

    UNSET(USER_FILE)
    UNSET(OUTPUT_PATH)
    UNSET(_CMAKE_LocalDebuggerEnvironment)
endif()
