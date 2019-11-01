# - Try to find SeetaFace
#
#  Usage from an external project:
#    In your CMakeLists.txt, add these lines:
#
#    find_package(SeetaFace)
#    target_link_libraries(MY_TARGET_NAME ${SeetaFace_LIBRARIES})
#  
#  This file will define the following variables:
#    SeetaFace_FOUND: True if find SeetaFace, other false
#    SeetaFace_LIBRARIES:  The list of all imported targets for SeetaFace components
#
# Author: Kang Lin <kl222@126.com>

include(FindPackageHandleStandardArgs)

if (NOT SeetaFace_FIND_COMPONENTS)
    set(SeetaFace_FIND_COMPONENTS
        SeetaNet
        SeetaFaceDetector
        SeetaFaceLandmarker
        SeetaFaceRecognizer
        SeetaFaceTracker
        SeetaQualityAssessor
	)
endif()

get_filename_component(_SeetaFace_module_paths "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

set(_SeetaFace_FIND_PARTS_REQUIRED)
if (SeetaFace_FIND_REQUIRED)
    set(_SeetaFace_FIND_PARTS_REQUIRED REQUIRED)
endif()
set(_SeetaFace_FIND_PARTS_QUIET)
if (SeetaFace_FIND_QUIETLY)
    set(_SeetaFace_FIND_PARTS_QUIET QUIET)
endif()

foreach(module ${SeetaFace_FIND_COMPONENTS})
    find_package(${module}
        ${_SeetaFace_FIND_PARTS_QUIET}
        ${_SeetaFace_FIND_PARTS_REQUIRED}
        PATHS ${_SeetaFace_module_paths} NO_DEFAULT_PATH
    )
    if(${module}_FOUND)
        list(APPEND SeetaFace_LIBRARIES SeetaFace::${module})
    endif()
    list(APPEND required "${module}_FOUND")  
endforeach()

# Run checks via find_package_handle_standard_args
find_package_handle_standard_args(SeetaFace
	FOUND_VAR SeetaFace_FOUND
	REQUIRED_VARS ${required})
