set(PADDLE_WITH_CUDA OFF)
if(CUDA_FOUND)
  set(PADDLE_WITH_CUDA ON)
  include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
endif(CUDA_FOUND)

message(STATUS "PADDLE_WITH_CUDA=" ${PADDLE_WITH_CUDA})
message(STATUS "CUDA_TOOLKIT_ROOT_DIR=" ${CUDA_TOOLKIT_ROOT_DIR})

add_subdirectory(fluid)
