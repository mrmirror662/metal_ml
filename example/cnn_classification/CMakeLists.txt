add_executable(cnn_classification main.cpp)
target_link_libraries(cnn_classification PRIVATE cg_metal mnist_common)
target_compile_options(cnn_classification PRIVATE -Wall -Wextra)
