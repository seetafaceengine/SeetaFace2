#
# Author: Kang Lin<kl222@126.com>
# Date  : 2019-08-20
#

cmake_minimum_required(VERSION 2.8)

project("search")

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
    ${CMAKE_SOURCE_DIR}/FaceRecognizer/include
    )
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    )

target_link_libraries(${PROJECT_NAME} PRIVATE
    seetanet2 SeetaFaceDetector2 SeetaFaceLandmarker2 SeetaFaceRecognizer2)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

INSTALL(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
