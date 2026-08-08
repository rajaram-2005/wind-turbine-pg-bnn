# Edge inference engine for the physics-guided BNN (ONNX Runtime, C++17).
#
# Build:
#   cmake -B build -DONNXRUNTIME_ROOT=/opt/onnxruntime
#   cmake --build build --config Release
#
# ONNXRUNTIME_ROOT must contain include/ (onnxruntime_cxx_api.h) and lib/.

cmake_minimum_required(VERSION 3.16)
project(pg_bnn_edge_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to the ONNX Runtime install prefix")

if(ONNXRUNTIME_ROOT STREQUAL "")
    # Fall back to common system locations / find_package config if provided.
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        PATHS /usr/local/include/onnxruntime /usr/include/onnxruntime
        PATH_SUFFIXES core/session)
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS /usr/local/lib /usr/lib)
else()
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
        "ONNX Runtime not found. Set -DONNXRUNTIME_ROOT=<prefix> to the "
        "extracted onnxruntime release (containing include/ and lib/).")
endif()

add_library(pg_bnn_inference STATIC inference.cpp)
target_include_directories(pg_bnn_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(pg_bnn_inference PUBLIC ${ONNXRUNTIME_LIB})

# Optional demo executable: pass -DBUILD_DEMO=ON and provide demo_main.cpp.
option(BUILD_DEMO "Build the demo CLI (requires demo_main.cpp)" OFF)
if(BUILD_DEMO)
    add_executable(pg_bnn_demo demo_main.cpp)
    target_link_libraries(pg_bnn_demo PRIVATE pg_bnn_inference)
endif()

install(TARGETS pg_bnn_inference ARCHIVE DESTINATION lib)
install(FILES inference.h DESTINATION include/pgbnn)
