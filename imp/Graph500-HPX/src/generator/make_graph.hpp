//  Copyright (c)      2014 Thomas Heller
//  Copyright (C) 2009-2010 The Trustees of Indiana University.
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Original Authors: Jeremiah Willcock
//                    Andrew Lumsdaine 

#include <hpx/config.hpp>
#include <hpx/config/export_definitions.hpp>
#include <generator/graph_generator.hpp>

#include <cstdint>
#include <vector>

namespace graph500 { namespace generator
{
    std::vector<packed_edge> HPX_EXPORT make_graph(
        int log_numverts
      , std::int64_t desired_nedges
      , std::uint64_t userseed1
      , std::uint64_t userseed2
      , std::size_t rank
      , std::size_t size
    );

    std::vector<double> HPX_EXPORT make_random_numbers(
        std::size_t nvalues
      , std::uint64_t userseed1
      , std::uint64_t userseed2
      , std::int64_t position
    );
}}
