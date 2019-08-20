#
# Author: Kang Lin<kl222@126.com>
# Date  : 2019-08-20
#

cmake_minimum_required(VERSION 2.8)

project("points81")

if(UNIX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
endif()

find_package(OpenCV REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
# add library
add_executable(${PROJECT_NAME} example.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/SeetaNet/include
    ${CMAKE_SOURCE_DIR}/FaceDetector/include
    ${CMAKE_SOURCE_DIR}/FaceLandmarker/include
    )
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    )

target_link_libraries(${PROJECT_NAME} PRIVATE
    seetanet2 SeetaFaceDetector2 SeetaFaceLandmarker2)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

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
