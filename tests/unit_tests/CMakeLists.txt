add_executable(unit_tests main.cpp)
target_link_libraries(unit_tests PRIVATE cg_cpu cg_metal)
target_include_directories(unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(unit_tests PRIVATE -Wall -Wextra)
