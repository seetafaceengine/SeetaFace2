#
# Author: Kang Lin<kl222@126.com>
# Date  : 2019-08-20
#

cmake_minimum_required(VERSION 2.8)
project(SeetaNet)

add_definitions(-DSEETA_EXPORTS)

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
    
# for compile
add_library(${PROJECT_NAME} ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/include_inner
    ${PROJECT_SOURCE_DIR}/src/include_inner/layers
    ${PROJECT_SOURCE_DIR}/src
    )
if(ANDROID_ARM_NEON)
    target_compile_definitions(${PROJECT_NAME} PUBLIC __ARM_NEON__)
    set(SeetaNet_definitions "-D__ARM_NEON__ ${SeetaNet_definitions}")
endif()

# Set flags
if(ANDROID)
    target_compile_options(${PROJECT_NAME} PRIVATE 
        -frtti -fexceptions -funsafe-math-optimizations -ftree-vectorize -ffast-math)
    target_link_options(${PROJECT_NAME} PUBLIC -fuse-ld=bfd)
endif()

GENERATE_EXPORT_HEADER(${PROJECT_NAME}
    EXPORT_MACRO_NAME SEETA_NET_API
    EXPORT_FILE_NAME SeetaNetExport.h
    )
file(COPY ${CMAKE_CURRENT_BINARY_DIR}/SeetaNetExport.h
    DESTINATION ${CMAKE_BINARY_DIR})
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/SeetaNetExport.h
  DESTINATION
    ${CMAKE_INSTALL_INCLUDEDIR}/seeta
)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_BINARY_DIR}
    )

# link library
if (NOT ANDROID AND UNIX) 
    target_link_libraries(${PROJECT_NAME} PUBLIC pthread)
    set(SeetaNet_LIBS "pthread ${SeetaNet_LIBS}")
endif()
if (WIN32) 
    target_link_libraries(${PROJECT_NAME} PUBLIC Ws2_32)
    set(SeetaNet_LIBS "Ws2_32 ${SeetaNet_LIBS}")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    VERSION ${BUILD_VERSION})
# Be will to install header files
set_property(TARGET ${PROJECT_NAME} PROPERTY PUBLIC_HEADER ${INCLUDE_FILES})

if(WIN32)
    INSTALL(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}Config
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_BINDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/seeta
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
else()
    INSTALL(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}Config
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/seeta
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
endif()
export(TARGETS ${PROJECT_NAME}
       APPEND FILE ${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake
)
# Install cmake configure files
install(
        EXPORT ${PROJECT_NAME}Config
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
        NAMESPACE
            SeetaFace::
)
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${BUILD_VERSION}
    COMPATIBILITY AnyNewerVersion)
install(FILES "${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake")
# Install configure file
set(SeetaFace2_LIBS "${PROJECT_NAME} ${SeetaFace2_LIBS}" PARENT_SCOPE)
configure_file(SeetaNet.pc.in ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

