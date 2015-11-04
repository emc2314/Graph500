//  Copyright (c) 2014 Thomas Heller
//  Copyright (C) 2010 The Trustees of Indiana University.
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Original Authors: Jeremiah Willcock
//                    Andrew Lumsdaine 

#include <cstdint>

#include <generator/splittable_mrg.hpp>

#include <iostream>

/* Multiple recursive generator from L'Ecuyer, P., Blouin, F., and       */
/* Couture, R. 1993. A search for good multiple recursive random number  */
/* generators. ACM Trans. Model. Comput. Simul. 3, 2 (Apr. 1993), 87-98. */
/* DOI= http://doi.acm.org/10.1145/169702.169698 -- particular generator */
/* used is from table 3, entry for m = 2^31 - 1, k = 5 (same generator   */
/* is used in GNU Scientific Library).                                   */
/*                                                                       */
/* MRG state is 5 numbers mod 2^31 - 1, and there is a transition matrix */
/* A from one state to the next:                                         */
/*                                                                       */
/* A = [x 0 0 0 y]                                                       */
/*     [1 0 0 0 0]                                                       */
/*     [0 1 0 0 0]                                                       */
/*     [0 0 1 0 0]                                                       */
/*     [0 0 0 1 0]                                                       */
/* where x = 107374182 and y = 104480                                    */
/*                                                                       */
/* To do leapfrogging (applying multiple steps at once so that we can    */
/* create a tree of generators), we need powers of A.  These (from an    */
/* analysis with Maple) all look like:                                   */
/*                                                                       */
/* let a = x * s + t                                                     */
/*     b = x * a + u                                                     */
/*     c = x * b + v                                                     */
/*     d = x * c + w in                                                  */
/* A^n = [d   s*y a*y b*y c*y]                                           */
/*       [c   w   s*y a*y b*y]                                           */
/*       [b   v   w   s*y a*y]                                           */
/*       [a   u   v   w   s*y]                                           */
/*       [s   t   u   v   w  ]                                           */
/* for some values of s, t, u, v, and w                                  */
/* Note that A^n is determined by its bottom row (and x and y, which are */
/* fixed), and that it has a large part that is a Toeplitz matrix.  You  */
/* can multiply two A-like matrices by:                                  */
/* (defining a..d1 and a..d2 for the two matrices)                       */
/* s3 = s1 d2 + t1 c2 + u1 b2 + v1 a2 + w1 s2,                           */
/* t3 = s1 s2 y + t1 w2 + u1 v2 + v1 u2 + w1 t2,                         */
/* u3 = s1 a2 y + t1 s2 y + u1 w2 + v1 v2 + w1 u2,                       */
/* v3 = s1 b2 y + t1 a2 y + u1 s2 y + v1 w2 + w1 v2,                     */
/* w3 = s1 c2 y + t1 b2 y + u1 a2 y + v1 s2 y + w1 w2                    */

namespace graph500 { namespace generator
{
    void mrg_state::step(transition_matrix const & m)
    {
        std::uint_fast32_t o1
            = mod_mac_y(mod_mul(m.d, z1), mod_mac4(0, m.s, z2, m.a, z3, m.b, z4, m.c, z5));
        std::uint_fast32_t o2
            = mod_mac_y(mod_mac2(0, m.c, z1, m.w, z2), mod_mac3(0, m.s, z3, m.a, z4, m.b, z5));
        std::uint_fast32_t o3
            = mod_mac_y(mod_mac3(0, m.b, z1, m.v, z2, m.w, z3), mod_mac2(0, m.s, z4, m.a, z5));
        std::uint_fast32_t o4
            = mod_mac_y(mod_mac4(0, m.a, z1, m.u, z2, m.v, z3, m.w, z4), mod_mul(m.s, z5));
        std::uint_fast32_t o5
            = mod_mac2(mod_mac3(0, m.s, z1, m.t, z2, m.u, z3), m.v, z4, m.w, z5);
          
        z1 = o1;
        z2 = o2;
        z3 = o3;
        z4 = o4;
        z5 = o5;
    }

    void mrg_state::orig_step()
    {
        std::uint_fast32_t new_elt = mod_mac_y(mod_mul_x(z1), z5);
        z5 = z4;
        z4 = z3;
        z3 = z2;
        z2 = z1;
        z1 = new_elt;
    }

    void mrg_state::skip(
        std::uint_least64_t exponent_high
      , std::uint_least64_t exponent_middle
      , std::uint_least64_t exponent_low
      )
    {
        for(std::size_t byte_index = 0; exponent_low; ++byte_index, exponent_low >>=8)
        {
            std::uint_least8_t val = static_cast<std::uint_least8_t>(exponent_low & 0xFF);
            if(val != 0)
            {
                step(transition_matrix::skip_matrices[byte_index][val]);
            }
        }
        for(std::size_t byte_index = 8; exponent_middle; ++byte_index, exponent_middle >>=8)
        {
            std::uint_least8_t val = static_cast<std::uint_least8_t>(exponent_middle & 0xFF);
            if(val != 0)
            {
                step(transition_matrix::skip_matrices[byte_index][val]);
            }
        }
        for(std::size_t byte_index = 16; exponent_high; ++byte_index, exponent_high >>=8)
        {
            std::uint_least8_t val = static_cast<std::uint_least8_t>(exponent_high & 0xFF);
            if(val != 0)
            {
                step(transition_matrix::skip_matrices[byte_index][val]);
            }
        }
    }
    
    std::uint_fast32_t mrg_state::get_uint_orig()
    {
        orig_step();
        return z1;
    }
    
    double mrg_state::get_double_orig()
    {
        return
            static_cast<double>(get_uint_orig()) * .000000000465661287524579692 /* (2^31 - 1)^(-1) */ +
            static_cast<double>(get_uint_orig()) * .0000000000000000002168404346990492787 /* (2^31 - 1)^(-2) */
        ;
    }

    mrg_state::mrg_state(seeds_type const & seed)
      : z1(seed[0])
      , z2(seed[1])
      , z3(seed[2])
      , z4(seed[3])
      , z5(seed[4])
    {
    }

}}
