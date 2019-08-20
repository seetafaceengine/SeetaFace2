#
# Author: Kang Lin<kl222@126.com>
# Date  : 2019-08-20
#

cmake_minimum_required(VERSION 2.8)

project(point81)

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
    ${CMAKE_SOURCE_DIR}/FaceDetector/include
    ${CMAKE_SOURCE_DIR}/FaceLandmarker/include
    )
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    )

target_link_libraries(${PROJECT_NAME} PRIVATE
    SeetaNet SeetaFaceDetector SeetaFaceLandmarker)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${BUILD_VERSION})

INSTALL(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
