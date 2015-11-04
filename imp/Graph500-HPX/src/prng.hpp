//  Copyright (c)      2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <generator/splittable_mrg.hpp>
#include <generator/utils.hpp>

#include <cstdint>

namespace graph500
{
    struct prng_state
    {
        static generator::seeds_type make_seed(std::uint64_t userseed)
        {
            generator::seeds_type seed;
            
            seed[0] = (userseed & 0x3FFFFFFF) + 1;
            seed[1] = ((userseed >> 30) & 0x3FFFFFFF) + 1;
            seed[2] = (userseed & 0x3FFFFFFF) + 1;
            seed[3] = ((userseed >> 30) & 0x3FFFFFFF) + 1;
            seed[4] = ((userseed >> 60) << 4) + (userseed >> 60) + 1;

            return seed;
        }

        prng_state()
            // TODO: read seed from env
          : userseed(0xDECAFBAD)
          , seed(make_seed(userseed))
          , state(seed)
        {
        }

        std::uint64_t userseed;
        generator::seeds_type seed;
        generator::mrg_state state;
    };
}
