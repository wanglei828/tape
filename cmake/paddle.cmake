# Copyright (c) 2018 Yi Wang <yi.wang.2005@gmail.com> All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

INCLUDE(ExternalProject)

SET(PADDLE_SOURCES_DIR ${CMAKE_SOURCE_DIR}/paddle)
SET(PADDLE_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/third_party/paddle)
SET(PADDLE_INCLUDE_DIR "${PADDLE_INSTALL_DIR}/include" CACHE PATH "paddle include directory." FORCE)

IF(WIN32)
    SET(PADDLE_LIBRARIES "${PADDLE_INSTALL_DIR}/lib/paddle.lib" CACHE FILEPATH "paddle library." FORCE)
ELSE(WIN32)
    SET(PADDLE_LIBRARIES "${PADDLE_INSTALL_DIR}/lib/libpaddle.a" CACHE FILEPATH "paddle library." FORCE)
ENDIF(WIN32)

INCLUDE_DIRECTORIES(${PADDLE_INCLUDE_DIR})

ExternalProject_Add(
    extern_paddle
    ${EXTERNAL_PROJECT_LOG_ARGS}
    DEPENDS gflags
    SOURCE_DIR       ${PADDLE_SOURCES_DIR}
    UPDATE_COMMAND   ""
    DOWNLOAD_COMMAND ""
    ${EXTERNAL_PROJECT_CMAKE_ARGS}
    CMAKE_ARGS       -DCMAKE_INSTALL_PREFIX=${PADDLE_INSTALL_DIR}
    CMAKE_ARGS       -DCMAKE_INSTALL_LIBDIR=${PADDLE_INSTALL_DIR}/lib
    CMAKE_ARGS       -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    CMAKE_ARGS       -DWITH_TESTING=OFF
    CMAKE_ARGS       -DWITH_PYTHON=OFF
    CMAKE_ARGS       -DWITH_GPU=OFF
    CMAKE_ARGS       -DWITH_FLUID_ONLY=ON
    CMAKE_ARGS       --target=operator
    CMAKE_ARGS       -DCMAKE_BUILD_TYPE=Release
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${PADDLE_INSTALL_DIR}
                     -DCMAKE_INSTALL_LIBDIR:PATH=${PADDLE_INSTALL_DIR}/lib
                     -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                     -DCMAKE_BUILD_TYPE:STRING=Release
)

ADD_LIBRARY(paddle STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET paddle PROPERTY IMPORTED_LOCATION ${PADDLE_LIBRARIES})
SET_PROPERTY(TARGET paddle PROPERTY INTERFACE_LINK_LIBRARIES gflags)
ADD_DEPENDENCIES(paddle extern_paddle gflags)

MESSAGE(STATUS "Google/paddle library: ${PADDLE_LIBRARIES}")
