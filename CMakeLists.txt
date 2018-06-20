cmake_minimum_required(VERSION 3.0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/bazel.cmake/cmake)
include(bazel)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/external)
include(snappy)
include(snappystream)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(paddle)
#add_subdirectory(tape)
