# library in subdir

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(UNIX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

add_definitions(-DSEETA_EXPORTS)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/seeta)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/seeta)

message(STATUS "install targets library00: " ${PROJECT_NAME})
#configure library version and name
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/seeta/${SeetaFaceDetector_NAME}Config.h @ONLY)

file(GLOB_RECURSE ROOT_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    )
	
	

file(GLOB_RECURSE ROOT_SEETA_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/seeta/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/seeta/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/seeta/*.cpp
    )

file(GLOB INCLUDE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/seeta/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/seeta/*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/seeta/*.hpp

    )

	
list(APPEND SRC_FILES ${ROOT_SRC_FILES})
list(APPEND SRC_FILES ${ROOT_SEETA_FILES})

# add library
add_library(${PROJECT_NAME} ${TARGET} ${SRC_FILES})



# set library properties
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${PROJECT_NAME}${ENV_SUFFIX})




# build dir
set(OUT_DIR ${SOLUTION_DIR}/build)

#create install dir
file(MAKE_DIRECTORY ${OUT_DIR})

set(LIBRARY_OUTPUT_PATH ${OUT_DIR}/${ENV_LIBRARY_DIR})
set(ARCHIVE_OUTPUT_PATH ${OUT_DIR}/${ENV_LIBRARY_DIR})

include_directories("${SEETANET_HOME}/include")

message(STATUS "SEETANET_HOME:${SEETANET_HOME}")
if(UNIX)
    target_link_libraries(${PROJECT_NAME} "${SEETANET_HOME}/${ENV_LIBRARY_DIR}/libseetanet2${ENV_SUFFIX}.so")
elseif(WIN32) 
    target_link_libraries(${PROJECT_NAME} "${SEETANET_HOME}/lib/${PLATFORM}/seetanet2${ENV_SUFFIX}.lib")  
endif()



# add command to copy header files
add_custom_command(TARGET ${PROJECT_NAME}
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}/${ENV_INCLUDE_DIR}
        )
add_custom_command(TARGET ${PROJECT_NAME}
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}/${ENV_INCLUDE_DIR}/seeta
        )

# copy every header files
foreach(header ${INCLUDE_FILES})
    add_custom_command(TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header} ${OUT_DIR}/${ENV_INCLUDE_DIR}/seeta
            )
endforeach()


message(STATUS "install targets library: " ${PROJECT_NAME})
# if (WIN32)
	# install(TARGETS ${PROJECT_NAME} DESTINATION ${ENV_LIBRARY_DIR})
# else()
	install(TARGETS ${PROJECT_NAME}
		RUNTIME DESTINATION ${ENV_RUNTIME_DIR}
		LIBRARY DESTINATION ${ENV_LIBRARY_DIR}
		ARCHIVE DESTINATION ${ENV_ARCHIVE_DIR}
	)
# endif()



