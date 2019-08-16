cmake_minimum_required(VERSION 2.8)
project(seetanet2)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# option for platform
#set(PLATFORM "x64" CACHE STRING "x86 or x64")
#set(CONFIGURATION "Release" CACHE STRING "Debug or Release")



# option for extern


# option for install
set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/build CACHE STRING "set install prefix" FORCE)


# env
include(LOCAL_ENV)

# check options
if (${PLATFORM} STREQUAL "x86")
    message(STATUS "Platform: " ${PLATFORM})
else()
    message(STATUS "Platform: " ${PLATFORM})
    add_definitions(-D_X64_)
    set(X64_ARCH ON)
endif()

#set(CONFIGURATION "Debug")
if (${CONFIGURATION} STREQUAL "Debug")
    message(STATUS "Configuration: " ${CONFIGURATION})
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "Configuration: " ${CONFIGURATION})
    set(CMAKE_BUILD_TYPE "Release")
endif()





# set flags
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
        set(CMAKE_C_FLAGS "-m32  -msse2 -mfma ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-m32 -msse2 -mfma ${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_C_FLAGS "-m64 -msse2 -mfma ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-m64 -msse2 -mfma ${CMAKE_CXX_FLAGS}")
    endif()
elseif(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()



# add headers
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src/include_inner)
include_directories(${PROJECT_SOURCE_DIR}/src/include_inner/layers)
include_directories(${PROJECT_SOURCE_DIR}/src)

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
ADD_LIBRARY(${PROJECT_NAME}_SHARED SHARED ${INCLUDE_FILES} ${SRC_INCLUDE_FILES} ${SRC_FILES})

set_target_properties(${PROJECT_NAME}_SHARED PROPERTIES OUTPUT_NAME ${PROJECT_NAME}${ENV_SUFFIX})


set(OUT_DIR ${PROJECT_SOURCE_DIR}/build)
#set(LIBRARY_OUTPUT_PATH ${OUT_DIR}/lib/${PLATFORM})

set(LIBRARY_OUTPUT_PATH ${OUT_DIR}/${ENV_LIBRARY_DIR})

# add command to copy header files
add_custom_command(TARGET ${PROJECT_NAME}_SHARED
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}/include
        )

# copy every header files
foreach(header ${INCLUDE_FILES})
    add_custom_command(TARGET ${PROJECT_NAME}_SHARED
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header} ${OUT_DIR}/include
            )
endforeach()



# link library
if (UNIX) 
    target_link_libraries(${PROJECT_NAME}_SHARED -lpthread)   
endif()

# install
install(TARGETS ${PROJECT_NAME}_SHARED DESTINATION ${CMAKE_INSTALL_PREFIX}/${ENV_LIBRARY_DIR})
install(FILES ${INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

