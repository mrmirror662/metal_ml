find_library(METAL_LIB      Metal)
find_library(FOUNDATION_LIB Foundation)
find_library(QUARTZCORE_LIB QuartzCore)
find_program(XCRUN           xcrun REQUIRED)

if(NOT METAL_LIB)
    message(STATUS "Metal not found — skipping cg_metal")
    return()
endif()

# --- Metal-CPP: auto-download to third_party/ if missing -------------------
set(METAL_CPP_DIR "${CMAKE_SOURCE_DIR}/third_party/metal-cpp")
set(METAL_CPP_URL "https://developer.apple.com/metal/cpp/files/metal-cpp_26.4.zip")

if(NOT EXISTS "${METAL_CPP_DIR}/Metal/Metal.hpp")
    set(METAL_CPP_ZIP "${CMAKE_SOURCE_DIR}/third_party/metal-cpp.zip")
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/third_party")
    if(NOT EXISTS "${METAL_CPP_ZIP}")
        message(STATUS "Downloading Metal-CPP from ${METAL_CPP_URL}")
        file(DOWNLOAD ${METAL_CPP_URL} ${METAL_CPP_ZIP}
             STATUS dl_status SHOW_PROGRESS)
        list(GET dl_status 0 dl_code)
        if(NOT dl_code EQUAL 0)
            file(REMOVE ${METAL_CPP_ZIP})
            list(GET dl_status 1 dl_msg)
            message(FATAL_ERROR "Metal-CPP download failed: ${dl_msg}")
        endif()
    endif()
    message(STATUS "Extracting Metal-CPP to ${METAL_CPP_DIR}")
    file(ARCHIVE_EXTRACT INPUT ${METAL_CPP_ZIP}
         DESTINATION "${CMAKE_SOURCE_DIR}/third_party")
endif()
set(SHADER_SRC    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/kernels.metal")
set(SHADER_AIR    "${CMAKE_CURRENT_BINARY_DIR}/kernels.air")
set(SHADER_LIB    "${CMAKE_CURRENT_BINARY_DIR}/kernels.metallib")

# .metal -> .air -> .metallib
add_custom_command(
    OUTPUT  ${SHADER_LIB}
    COMMAND ${XCRUN} -sdk macosx metal -c ${SHADER_SRC} -o ${SHADER_AIR}
    COMMAND ${XCRUN} -sdk macosx metallib ${SHADER_AIR} -o ${SHADER_LIB}
    DEPENDS ${SHADER_SRC}
    COMMENT "Compiling Metal shaders"
)
add_custom_target(cg_metal_shaders DEPENDS ${SHADER_LIB})

# Bake the metallib path in so the executor can load it at runtime
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/shader_path.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/shader_path.h"
)

add_library(cg_metal STATIC
    metal_executor.cpp
)
add_dependencies(cg_metal cg_metal_shaders)

target_include_directories(cg_metal
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    PRIVATE
        ${METAL_CPP_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}  # for generated shader_path.h
)

target_link_libraries(cg_metal
    PUBLIC  cg
    PRIVATE ${METAL_LIB} ${FOUNDATION_LIB} ${QUARTZCORE_LIB}
)
