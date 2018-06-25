cc_library(ddim SRCS ddim.cc DEPS eigen3 boost)
cc_test(ddim_test SRCS ddim_test.cc DEPS ddim)
nv_test(dim_test SRCS dim_test.cu DEPS ddim)

proto_library(framework_proto SRCS framework.proto)
