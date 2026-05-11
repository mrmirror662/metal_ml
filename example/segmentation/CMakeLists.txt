# Segmentation example: pretrained U-Net for human/person segmentation,
# loaded from .safetensors, run on Metal, output mask PNGs.
#
# Configure-time downloads (cached after first run, no network thereafter):
#   - human_unet.pth         (FiniUdesa/unet-human-segmentation, ~98 MB)
#   - human_unet.safetensors (converted from the .pth via a one-shot uv-script)
#   - stb_image.h + stb_image_write.h
#   - sample images of people
#
# Outputs (mask.png, overlay.png, input.png) are written into
# example/segmentation/outputs/ — visible directly in the source tree.

set(SEG_DATA_DIR    "${CMAKE_CURRENT_BINARY_DIR}/data")
set(SEG_SAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/samples")
set(SEG_OUT_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/outputs")
file(MAKE_DIRECTORY ${SEG_DATA_DIR})
file(MAKE_DIRECTORY ${SEG_SAMPLES_DIR})
file(MAKE_DIRECTORY ${SEG_OUT_DIR})

# ---- Pretrained weights ----------------------------------------------------
# Source: FiniUdesa/unet-human-segmentation on HuggingFace.
# Architecture: segmentation_models_pytorch.Unet, encoder=resnet34,
# decoder_channels=[256,128,64,32,16], classes=1 (binary mask).
# Trained on mattmdjaga/human_parsing_dataset. Dice 0.956 / IoU 0.916.
# Distributed as a PyTorch pickle; we convert to .safetensors at configure time.
set(SEG_PTH_URL  "https://huggingface.co/FiniUdesa/unet-human-segmentation/resolve/main/pytorch_model.pth")
set(SEG_PTH_PATH "${SEG_DATA_DIR}/human_unet.pth")
set(SEG_WEIGHTS_PATH "${SEG_DATA_DIR}/human_unet.safetensors")

if(NOT EXISTS ${SEG_PTH_PATH})
    message(STATUS "Downloading human U-Net .pth (~98 MB) from ${SEG_PTH_URL}")
    file(DOWNLOAD ${SEG_PTH_URL} ${SEG_PTH_PATH} STATUS dl_status SHOW_PROGRESS)
    list(GET dl_status 0 dl_code)
    if(NOT dl_code EQUAL 0)
        file(REMOVE ${SEG_PTH_PATH})
        list(GET dl_status 1 dl_msg)
        message(FATAL_ERROR ".pth download failed: ${dl_msg}")
    endif()
endif()

if(NOT EXISTS ${SEG_WEIGHTS_PATH})
    find_program(UV_BIN uv)
    if(NOT UV_BIN)
        message(FATAL_ERROR
            "uv not found on PATH — needed for the one-shot .pth -> .safetensors\n"
            "conversion. Install: curl -LsSf https://astral.sh/uv/install.sh | sh")
    endif()
    message(STATUS "Converting .pth -> .safetensors")
    execute_process(
        COMMAND ${UV_BIN} run --script
                "${CMAKE_CURRENT_SOURCE_DIR}/convert_to_safetensors.py"
                --input  "${SEG_PTH_PATH}"
                --output "${SEG_WEIGHTS_PATH}"
        RESULT_VARIABLE conv_rc OUTPUT_VARIABLE conv_out ERROR_VARIABLE conv_err)
    if(NOT conv_rc EQUAL 0)
        file(REMOVE ${SEG_WEIGHTS_PATH})
        message(FATAL_ERROR "Conversion failed:\n${conv_out}\n${conv_err}")
    endif()
endif()

# ---- stb single-header image I/O -------------------------------------------
set(STB_DIR     "${CMAKE_SOURCE_DIR}/third_party/stb")
set(STB_COMMIT  "f0569113c93ad095470c54bf34a17b36646bbbb5")
file(MAKE_DIRECTORY ${STB_DIR})

foreach(hdr "stb_image.h" "stb_image_write.h")
    set(target "${STB_DIR}/${hdr}")
    if(NOT EXISTS ${target})
        message(STATUS "Downloading ${hdr}")
        file(DOWNLOAD "https://raw.githubusercontent.com/nothings/stb/${STB_COMMIT}/${hdr}"
             ${target} STATUS dl_status)
        list(GET dl_status 0 dl_code)
        if(NOT dl_code EQUAL 0)
            file(REMOVE ${target})
            list(GET dl_status 1 dl_msg)
            message(FATAL_ERROR "${hdr} download failed: ${dl_msg}")
        endif()
    endif()
endforeach()

# ---- Sample input images (downloaded once into the source tree so they show
#      up in editors, and so the default input is right there next to the code).
set(SEG_SAMPLES
    "person_portrait.jpg|https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=800"
    "person_full.jpg|https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=800"
    "person_group.jpg|https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=800"
    "person_walking.jpg|https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=800"
    "person_yoga.jpg|https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=800"
    "person_candid.jpg|https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=800"
)
foreach(entry IN LISTS SEG_SAMPLES)
    string(REPLACE "|" ";" parts ${entry})
    list(GET parts 0 fn)
    list(GET parts 1 url)
    set(target "${SEG_SAMPLES_DIR}/${fn}")
    if(NOT EXISTS ${target})
        message(STATUS "Downloading sample ${fn}")
        file(DOWNLOAD ${url} ${target} STATUS dl_status)
        list(GET dl_status 0 dl_code)
        if(NOT dl_code EQUAL 0)
            file(REMOVE ${target})
            message(WARNING "sample ${fn} download failed; skipping")
        endif()
    endif()
endforeach()

set(SEG_INPUT_PATH "${SEG_SAMPLES_DIR}/person_portrait.jpg")

# ---- Generated paths header ------------------------------------------------
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/seg_paths.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/seg_paths.h"
    @ONLY)

# ---- Target ----------------------------------------------------------------
add_executable(segmentation main.cpp)
target_include_directories(segmentation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${STB_DIR})
target_link_libraries(segmentation PRIVATE cg_metal)
target_compile_options(segmentation PRIVATE -Wall -Wextra -Wno-deprecated-declarations)
