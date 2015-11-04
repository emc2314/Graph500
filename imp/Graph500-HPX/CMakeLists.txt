# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

project(graph500)

cmake_minimum_required(VERSION 2.8)

if(NOT HPX_ROOT)
    if(NOT $ENV{HPX_ROOT} STREQUAL "")
        set(HPX_ROOT $ENV{HPX_ROOT})
   else()
       message(FATAL_ERROR "graph500 needs HPX. Please set HPX_ROOT.")
   endif()
endif()

if(HPX_ROOT)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${HPX_ROOT}/share/cmake-2.8/Modules)
endif()

################################################################################
# enable solution folders for MSVC
if(MSVC)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()
################################################################################

find_package(HPX REQUIRED)

include_directories(src)
include_directories(${HPX_INCLUDE_DIR})

set(HPX_RPATH "${CMAKE_INSTALL_PREFIX}/lib/hpx:${CMAKE_BINARY_DIR}/lib/hpx")
link_directories(${CMAKE_BINARY_DIR}/lib/hpx)
link_directories(${HPX_LIBRARY_DIR})

add_subdirectory(src)
