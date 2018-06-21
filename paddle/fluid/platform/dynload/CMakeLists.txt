cc_library(dynamic_loader SRCS dynamic_loader.cc DEPS glog gflags enforce)

list(APPEND CUDA_SRCS cublas.cc cudnn.cc curand.cc nccl.cc)
if (TENSORRT_FOUND)
  list(APPEND CUDA_SRCS tensorrt.cc)
endif()

nv_library(dynload_cuda SRCS ${CUDA_SRCS} DEPS dynamic_loader)

