/* Copyright (C) 2010 The Trustees of Indiana University.                  */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#ifndef GRAPH500_GENERATOR_MOD_ARITH_64BIT_HPP
#define GRAPH500_GENERATOR_MOD_ARITH_64BIT_HPP

#include <cstdint>
#include <hpx/util/assert.hpp>

namespace graph500 { namespace generator
{

    /* Various modular arithmetic operations for modulus 2^31-1 (0x7FFFFFFF).
     * These may need to be tweaked to get acceptable performance on some platforms
     * (especially ones without conditional moves). */

    inline std::uint_fast32_t mod_add(std::uint_fast32_t a, std::uint_fast32_t b) {
      std::uint_fast32_t x;
      HPX_ASSERT (a <= 0x7FFFFFFE);
      HPX_ASSERT (b <= 0x7FFFFFFE);
#if 0
      return (a + b) % 0x7FFFFFFF;
#else
      x = a + b; /* x <= 0xFFFFFFFC */
      x = (x >= 0x7FFFFFFF) ? (x - 0x7FFFFFFF) : x;
      return x;
#endif
    }

    inline std::uint_fast32_t mod_mul(std::uint_fast32_t a, std::uint_fast32_t b) {
      std::uint_fast64_t temp;
      std::uint_fast32_t temp2;
      HPX_ASSERT (a <= 0x7FFFFFFE);
      HPX_ASSERT (b <= 0x7FFFFFFE);
#if 0
      return (std::uint_fast32_t)((std::uint_fast64_t)a * b % 0x7FFFFFFF);
#else
      temp = (std::uint_fast64_t)a * b; /* temp <= 0x3FFFFFFE00000004 */
      temp2 = (std::uint_fast32_t)(temp & 0x7FFFFFFF) + (std::uint_fast32_t)(temp >> 31); /* temp2 <= 0xFFFFFFFB */
      return (temp2 >= 0x7FFFFFFF) ? (temp2 - 0x7FFFFFFF) : temp2;
#endif
    }

    inline std::uint_fast32_t mod_mac(std::uint_fast32_t sum, std::uint_fast32_t a, std::uint_fast32_t b) {
      std::uint_fast64_t temp;
      std::uint_fast32_t temp2;
      HPX_ASSERT (sum <= 0x7FFFFFFE);
      HPX_ASSERT (a <= 0x7FFFFFFE);
      HPX_ASSERT (b <= 0x7FFFFFFE);
#if 0
      return (std::uint_fast32_t)(((std::uint_fast64_t)a * b + sum) % 0x7FFFFFFF);
#else
      temp = (std::uint_fast64_t)a * b + sum; /* temp <= 0x3FFFFFFE80000002 */
      temp2 = (std::uint_fast32_t)(temp & 0x7FFFFFFF) + (std::uint_fast32_t)(temp >> 31); /* temp2 <= 0xFFFFFFFC */
      return (temp2 >= 0x7FFFFFFF) ? (temp2 - 0x7FFFFFFF) : temp2;
#endif
    }

    inline std::uint_fast32_t mod_mac2(std::uint_fast32_t sum, std::uint_fast32_t a, std::uint_fast32_t b, std::uint_fast32_t c, std::uint_fast32_t d) {
      HPX_ASSERT (sum <= 0x7FFFFFFE);
      HPX_ASSERT (a <= 0x7FFFFFFE);
      HPX_ASSERT (b <= 0x7FFFFFFE);
      HPX_ASSERT (c <= 0x7FFFFFFE);
      HPX_ASSERT (d <= 0x7FFFFFFE);
      return mod_mac(mod_mac(sum, a, b), c, d);
    }

    inline std::uint_fast32_t mod_mac3(std::uint_fast32_t sum, std::uint_fast32_t a, std::uint_fast32_t b, std::uint_fast32_t c, std::uint_fast32_t d, std::uint_fast32_t e, std::uint_fast32_t f) {
      HPX_ASSERT (sum <= 0x7FFFFFFE);
      HPX_ASSERT (a <= 0x7FFFFFFE);
      HPX_ASSERT (b <= 0x7FFFFFFE);
      HPX_ASSERT (c <= 0x7FFFFFFE);
      HPX_ASSERT (d <= 0x7FFFFFFE);
      HPX_ASSERT (e <= 0x7FFFFFFE);
      HPX_ASSERT (f <= 0x7FFFFFFE);
      return mod_mac2(mod_mac(sum, a, b), c, d, e, f);
    }

    inline std::uint_fast32_t mod_mac4(std::uint_fast32_t sum, std::uint_fast32_t a, std::uint_fast32_t b, std::uint_fast32_t c, std::uint_fast32_t d, std::uint_fast32_t e, std::uint_fast32_t f, std::uint_fast32_t g, std::uint_fast32_t h) {
      HPX_ASSERT (sum <= 0x7FFFFFFE);
      HPX_ASSERT (a <= 0x7FFFFFFE);
      HPX_ASSERT (b <= 0x7FFFFFFE);
      HPX_ASSERT (c <= 0x7FFFFFFE);
      HPX_ASSERT (d <= 0x7FFFFFFE);
      HPX_ASSERT (e <= 0x7FFFFFFE);
      HPX_ASSERT (f <= 0x7FFFFFFE);
      HPX_ASSERT (g <= 0x7FFFFFFE);
      HPX_ASSERT (h <= 0x7FFFFFFE);
      return mod_mac2(mod_mac2(sum, a, b, c, d), e, f, g, h);
    }

    /* The two constants x and y are special cases because they are easier to
     * multiply by on 32-bit systems.  They are used as multipliers in the random
     * number generator.  The techniques for fast multiplication by these
     * particular values are from:
     *
     * Pierre L'Ecuyer, Francois Blouin, and Raymond Couture. 1993. A search
     * for good multiple recursive random number generators. ACM Trans. Model.
     * Comput. Simul. 3, 2 (April 1993), 87-98. DOI=10.1145/169702.169698
     * http://doi.acm.org/10.1145/169702.169698
     *
     * Pierre L'Ecuyer. 1990. Random numbers for simulation. Commun. ACM 33, 10
     * (October 1990), 85-97. DOI=10.1145/84537.84555
     * http://doi.acm.org/10.1145/84537.84555
     */

    inline std::uint_fast32_t mod_mul_x(std::uint_fast32_t a) {
      static const std::int32_t q = 20 /* UINT32_C(0x7FFFFFFF) / 107374182 */;
      static const std::int32_t r = 7  /* UINT32_C(0x7FFFFFFF) % 107374182 */;
      std::int_fast32_t result = (int_fast32_t)(a) / q;
      result = 107374182 * ((int_fast32_t)(a) - result * q) - result * r;
      result += (result < 0 ? 0x7FFFFFFF : 0);
      HPX_ASSERT ((std::uint_fast32_t)(result) == mod_mul(a, 107374182));
      return (std::uint_fast32_t)result;
    }

    inline std::uint_fast32_t mod_mul_y(std::uint_fast32_t a) {
      static const std::int32_t q = 20554 /* UINT32_C(0x7FFFFFFF) / 104480 */;
      static const std::int32_t r = 1727  /* UINT32_C(0x7FFFFFFF) % 104480 */;
      std::int_fast32_t result = (int_fast32_t)(a) / q;
      result = 104480 * ((int_fast32_t)(a) - result * q) - result * r;
      result += (result < 0 ? 0x7FFFFFFF : 0);
      HPX_ASSERT ((std::uint_fast32_t)(result) == mod_mul(a, 104480));
      return (std::uint_fast32_t)result;
    }

    inline std::uint_fast32_t mod_mac_y(std::uint_fast32_t sum, std::uint_fast32_t a) {
      std::uint_fast32_t result = mod_add(sum, mod_mul_y(a));
      HPX_ASSERT (result == mod_mac(sum, a, 104480));
      return result;
    }
}}

#endif /* GRAPH500_GENERATOR_MOD_ARITH_32BIT_HPP */

