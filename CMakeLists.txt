cmake_minimum_required(VERSION 3.24)

project(
    rt2d
    VERSION 0.1.0
    LANGUAGES C CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

option(RT2D_USE_FETCH_DEPS "Fetch optional GitHub dependencies (fmt, glm)." OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_PREFIX_RESULT
    ERROR_VARIABLE TORCH_PREFIX_ERROR
)

if(NOT TORCH_PREFIX_RESULT EQUAL 0 OR TORCH_CMAKE_PREFIX_PATH STREQUAL "")
    message(
        FATAL_ERROR
        "Could not locate PyTorch CMake files using Python interpreter:\n"
        "  ${Python_EXECUTABLE}\n"
        "Install PyTorch in this interpreter, for example:\n"
        "  ${Python_EXECUTABLE} -m pip install torch\n"
        "Original error:\n${TORCH_PREFIX_ERROR}"
    )
endif()

list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pathlib, torch; print((pathlib.Path(torch.__file__).resolve().parent / 'lib').as_posix())"
    OUTPUT_VARIABLE TORCH_PY_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(RT2D_USE_FETCH_DEPS)
    include(FetchContent)

    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 11.0.2
    )
    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG 1.0.1
    )

    FetchContent_MakeAvailable(fmt glm)
endif()

add_library(
    rt2d_torch SHARED
    src/c/c_scene.c
    src/cpp/ops.cpp
    src/cpp/raytrace_cpu.cpp
    src/cuda/raytrace_cuda.cu
)

target_include_directories(rt2d_torch PRIVATE include ${Python_INCLUDE_DIRS})
if(Python_LIBRARY_DIRS)
    target_link_directories(rt2d_torch PRIVATE ${Python_LIBRARY_DIRS})
endif()
target_link_libraries(rt2d_torch PRIVATE ${TORCH_LIBRARIES})

if(TARGET fmt::fmt)
    target_link_libraries(rt2d_torch PRIVATE fmt::fmt)
    target_compile_definitions(rt2d_torch PRIVATE RT2D_USE_FMT=1)
endif()

if(TARGET glm::glm)
    target_link_libraries(rt2d_torch PRIVATE glm::glm)
    target_compile_definitions(rt2d_torch PRIVATE RT2D_USE_GLM=1)
endif()

target_compile_definitions(rt2d_torch PRIVATE TORCH_API_INCLUDE_EXTENSION_H)
target_compile_options(rt2d_torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

if(MSVC)
    target_compile_options(
        rt2d_torch
        PRIVATE
        $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:/W4>
    )
    target_compile_options(rt2d_torch PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/wd4267>)
else()
    target_compile_options(rt2d_torch PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(TORCH_CXX_FLAGS)
    separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(rt2d_torch PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${TORCH_CXX_FLAGS_LIST}>)
endif()

set_target_properties(
    rt2d_torch
    PROPERTIES
    OUTPUT_NAME rt2d_torch
)

install(
    TARGETS rt2d_torch
    RUNTIME DESTINATION python/rt2d
    LIBRARY DESTINATION python/rt2d
    ARCHIVE DESTINATION lib
)

if(WIN32)
    file(GLOB TORCH_DLLS "${TORCH_PY_LIB_DIR}/*.dll")
    foreach(TORCH_DLL IN LISTS TORCH_DLLS)
        add_custom_command(
            TARGET rt2d_torch POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TORCH_DLL}" "$<TARGET_FILE_DIR:rt2d_torch>"
        )
    endforeach()
    if(TORCH_DLLS)
        install(FILES ${TORCH_DLLS} DESTINATION python/rt2d)
    endif()
endif()

message(STATUS "PyTorch CMake prefix: ${TORCH_CMAKE_PREFIX_PATH}")
message(STATUS "PyTorch runtime lib dir: ${TORCH_PY_LIB_DIR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
