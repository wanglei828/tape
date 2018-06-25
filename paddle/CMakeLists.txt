set(PADDLE_WITH_CUDA OFF)
if(CUDA_FOUND)
  if(WITH_GPU)
    add_definitions(-DPADDLE_WITH_CUDA)
    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
  endif()
endif()

add_subdirectory(fluid)
