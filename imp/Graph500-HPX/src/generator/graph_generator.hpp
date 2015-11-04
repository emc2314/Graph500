//  Copyright (c) 2014 Thomas Heller
//  Copyright (C) 2009-2010 The Trustees of Indiana University.
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Original Authors: Jeremiah Willcock
//                    Andrew Lumsdaine 

#ifndef GRAPH500_GENERATOR_GRAPH_GENERATOR_HPP
#define GRAPH500_GENERATOR_GRAPH_GENERATOR_HPP

#include <generator/user_settings.hpp>
#include <generator/utils.hpp>

#include <cstdint>
#include <vector>

namespace graph500 { namespace generator
{
#ifdef GENERATOR_USE_PACKED_EDGE_TYPE
    struct packed_edge
    {
        std::uint32_t v0_low;
        std::uint32_t v1_low;
        std::uint32_t high;

        packed_edge(std::int64_t v0, std::int64_t v1)
          : v0_low(static_cast<std::uint32_t>(v0))
          , v1_low(static_cast<std::uint32_t>(v1))
          , high(static_cast<std::uint32_t>(((v0 >> 32) & 0xFFFF) | (((v1 >> 32) & 0xFFFF) << 16)))
        {}

        std::int64_t v0() const
        {
            return
                (v0_low
              | static_cast<std::int64_t>(
                    static_cast<std::int64_t>(
                        high & 0xFFFF
                    )
                 << 32
                ));
        }

        std::int64_t v1() const
        {
            return
                (v1_low
              | static_cast<std::int64_t>(
                    static_cast<std::int64_t>(
                        high >> 16
                    )
                 << 32
                ));
        }
    };
#else
    struct packed_edge
    {
        std::int64_t v0_;
        std::int64_t v1_;

        packed_edge(std::int64_t v0, std::int64_t v1)
          : v0_(v0)
          , v1_(v1)
        {}

        std::int64_t v0() const
        {
            return v0_;
        }

        std::int64_t v1() const
        {
            return v1_;
        }
    };
#endif

    std::vector<packed_edge> generate_kronecker_range(
        seeds_type const & seed
      , int logN
      , std::int64_t start_edge
      , std::int64_t end_edge
    );

}}

#endif
