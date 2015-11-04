//  Copyright (c) 2014 Thomas Heller
//  Copyright (C) 2010 The Trustees of Indiana University.
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Original Authors: Jeremiah Willcock
//                    Andrew Lumsdaine 

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif

#include <generator/utils.hpp>

namespace graph500 { namespace generator
{
    /*
    std::uint_fast64_t random_up_to(mrg_state & st, std::uint_fast64_t n)
    {
    }
    */

    seeds_type make_mrg_seed(std::uint64_t userseed1, std::uint64_t userseed2)
    {
        seeds_type seed;

        seed[0] = static_cast<uint32_t>(userseed1 & UINT32_C(0x3FFFFFFF)) + 1;
        seed[1] = static_cast<uint32_t>((userseed1 >> 30) & UINT32_C(0x3FFFFFFF)) + 1;
        seed[2] = static_cast<uint32_t>(userseed2 & UINT32_C(0x3FFFFFFF)) + 1;
        seed[3] = static_cast<uint32_t>((userseed2 >> 30) & UINT32_C(0x3FFFFFFF)) + 1;
        seed[4] = static_cast<uint32_t>((userseed2 >> 60) << 4) + static_cast<uint32_t>(userseed1 >> 60) + 1;

        return seed;
    }
}}
