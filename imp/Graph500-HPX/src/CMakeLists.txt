# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_library(
    graph500_generator
    ESSENTIAL
    SOURCES
        generator/splittable_mrg.cpp
        generator/transitions.cpp
        generator/graph_generator.cpp
        generator/make_graph.cpp
        generator/utils.cpp
    HEADERS
        generator/splittable_mrg.hpp
        generator/mod_arith.hpp
        generator/mod_arith_64bit.hpp
        generator/mod_arith_32bit.hpp
        generator/user_settings.hpp
    COMPILE_FLAGS
    LINK_FLAGS
    DEPENDENCIES
    FOLDER "generator"
)

add_hpx_executable(
    graph500
    ESSENTIAL
    SOURCES
        createpartitions.cpp
        main.cpp
        partition.cpp
    HEADERS
        createpartitions.hpp
        options.hpp
        partition.hpp
    COMPILE_FLAGS
    LINK_FLAGS
    DEPENDENCIES
    FOLDER "graph500"
)

add_hpx_executable(
    make_edgelist
    ESSENTIAL
    SOURCES make_edgelist.cpp
    HEADERS
    COMPILE_FLAGS
    LINK_FLAGS
    DEPENDENCIES
        graph500_generator
    FOLDER "make_edgelist"
)
