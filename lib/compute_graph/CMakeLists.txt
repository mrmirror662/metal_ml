add_library(cg STATIC
    cg.cpp
)

target_include_directories(cg
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

target_compile_options(cg PRIVATE
    -Wall -Wextra -Wpedantic
)

add_subdirectory(backends/cpu)
add_subdirectory(backends/metal)
