add_subdirectory(detail)

cc_library(malloc SRCS malloc.cc DEPS gpu_info buddy_allocator place enforce)

cc_library(memcpy SRCS memcpy.cc DEPS place)

cc_library(memory DEPS malloc memcpy)

cc_test(malloc_test SRCS malloc_test.cc DEPS malloc gtest)

nv_test(pinned_memory_test SRCS pinned_memory_test.cu DEPS place memory gtest)
