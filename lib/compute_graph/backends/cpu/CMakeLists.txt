add_library(cg_cpu STATIC
    cpu.cpp
)

target_include_directories(cg_cpu
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(cg_cpu PUBLIC cg)

target_compile_options(cg_cpu PRIVATE
    -Wall -Wextra -Wpedantic
)
