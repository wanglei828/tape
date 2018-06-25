cc_library(stringpiece SRCS piece.cc)
cc_test(stringpiece_test SRCS piece_test.cc DEPS stringpiece glog gflags gtest)
cc_test(stringprintf_test SRCS printf_test.cc DEPS glog gflags gtest)
cc_test(to_string_test SRCS to_string_test.cc DEPS gtest)
