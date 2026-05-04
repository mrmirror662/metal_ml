add_executable(mlp_classification main.cpp)
target_link_libraries(mlp_classification PRIVATE cg_cpu cg_metal mnist_common)
target_compile_options(mlp_classification PRIVATE -Wall -Wextra)
