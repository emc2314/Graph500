//  Copyright (c) 2014 Thomas Heller
//  Copyright (C) 2010 The Trustees of Indiana University.
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Original Authors: Jeremiah Willcock
//                    Andrew Lumsdaine

#ifndef GRAPH500_GENERATOR_UTILS_HPP
#define GRAPH500_GENERATOR_UTILS_HPP

#include <array>
#include <cstdint>

namespace graph500 { namespace generator
{
    typedef std::array<std::uint_fast32_t, 5> seeds_type;

    //std::uint_fast64_t random_up_to(mrg_state & st, std::uint_fast64_t n);
    seeds_type make_mrg_seed(std::uint64_t userseed1, std::uint64_t userseed2);
}}

#endif
