//  Copyright (c) 2014 Thomas Heller
//  Copyright (C) 2010 The Trustees of Indiana University.
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Original Authors: Jeremiah Willcock
//                    Andrew Lumsdaine 

#ifndef GRAPH500_GENERATOR_SPLITTABLE_MRG_HPP
#define GRAPH500_GENERATOR_SPLITTABLE_MRG_HPP

#include <generator/mod_arith.hpp>
#include <generator/utils.hpp>

namespace graph500 { namespace generator
{
    struct transition_matrix
    {
        std::uint_fast32_t s, t, u, v, w;
        std::uint_fast32_t a, b, c, d;

        static const transition_matrix skip_matrices[][256];
    };
    
    struct mrg_state
    {
        std::uint_fast32_t z1, z2, z3, z4, z5;
        void skip(
            std::uint_least64_t exponent_high
          , std::uint_least64_t exponent_middle
          , std::uint_least64_t exponent_low
        );
        
        std::uint_fast32_t get_uint_orig();
        double get_double_orig();

        mrg_state(seeds_type const & seed);

    private:
        void step(transition_matrix const & m);
        void orig_step();
    };
}}

#endif
