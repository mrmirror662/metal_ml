add_executable(app
    main.cpp
)

target_link_libraries(app PRIVATE cg_cpu cg_metal)

target_compile_options(app PRIVATE
    -Wall -Wextra -Wpedantic
)
