# Shared MNIST data + loader header. Both mlp_classification and
# cnn_classification link against `mnist_common` to get the dataset and the
# `mnist::` API.

set(MNIST_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
set(MNIST_FILES
    "train-images-idx3-ubyte"
    "train-labels-idx1-ubyte"
    "t10k-images-idx3-ubyte"
    "t10k-labels-idx1-ubyte"
)

foreach(name IN LISTS MNIST_FILES)
    set(target "${MNIST_DIR}/${name}")
    if(NOT EXISTS ${target})
        file(MAKE_DIRECTORY ${MNIST_DIR})
        message(STATUS "Downloading MNIST: ${name}.gz")
        file(DOWNLOAD
            "https://ossci-datasets.s3.amazonaws.com/mnist/${name}.gz"
            "${target}.gz"
            STATUS dl_status SHOW_PROGRESS
        )
        list(GET dl_status 0 dl_code)
        if(NOT dl_code EQUAL 0)
            file(REMOVE ${target}.gz)
            message(FATAL_ERROR "MNIST download failed")
        endif()
        execute_process(COMMAND gunzip -f ${target}.gz RESULT_VARIABLE rv)
        if(NOT rv EQUAL 0)
            message(FATAL_ERROR "gunzip failed for ${target}.gz")
        endif()
    endif()
endforeach()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/dataset_path.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/dataset_path.h
)

add_library(mnist_common INTERFACE)
target_include_directories(mnist_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}      # for "mnist.h"
    ${CMAKE_CURRENT_BINARY_DIR}      # for generated "dataset_path.h"
)
target_link_libraries(mnist_common INTERFACE cg)
