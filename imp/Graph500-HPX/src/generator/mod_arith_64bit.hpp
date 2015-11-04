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
      HPX_ASSERT (a <= 0x7FFFFFFE);
      HPX_ASSERT (b <= 0x7FFFFFFE);
      return (a + b) % 0x7FFFFFFF;
    }

    inline std::uint_fast32_t mod_mul(std::uint_fast32_t a, std::uint_fast32_t b) {
      HPX_ASSERT (a <= 0x7FFFFFFE);
      HPX_ASSERT (b <= 0x7FFFFFFE);
      return (std::uint_fast32_t)((std::uint_fast64_t)a * b % 0x7FFFFFFF);
    }

    inline std::uint_fast32_t mod_mac(std::uint_fast32_t sum, std::uint_fast32_t a, std::uint_fast32_t b) {
      HPX_ASSERT (sum <= 0x7FFFFFFE);
      HPX_ASSERT (a <= 0x7FFFFFFE);
      HPX_ASSERT (b <= 0x7FFFFFFE);
      return (std::uint_fast32_t)(((std::uint_fast64_t)a * b + sum) % 0x7FFFFFFF);
    }

    inline std::uint_fast32_t mod_mac2(std::uint_fast32_t sum, std::uint_fast32_t a, std::uint_fast32_t b, std::uint_fast32_t c, std::uint_fast32_t d) {
      HPX_ASSERT (sum <= 0x7FFFFFFE);
      HPX_ASSERT (a <= 0x7FFFFFFE);
      HPX_ASSERT (b <= 0x7FFFFFFE);
      HPX_ASSERT (c <= 0x7FFFFFFE);
      HPX_ASSERT (d <= 0x7FFFFFFE);
      return (std::uint_fast32_t)(((std::uint_fast64_t)a * b + (std::uint_fast64_t)c * d + sum) % 0x7FFFFFFF);
    }

    inline std::uint_fast32_t mod_mac3(std::uint_fast32_t sum, std::uint_fast32_t a, std::uint_fast32_t b, std::uint_fast32_t c, std::uint_fast32_t d, std::uint_fast32_t e, std::uint_fast32_t f) {
      HPX_ASSERT (sum <= 0x7FFFFFFE);
      HPX_ASSERT (a <= 0x7FFFFFFE);
      HPX_ASSERT (b <= 0x7FFFFFFE);
      HPX_ASSERT (c <= 0x7FFFFFFE);
      HPX_ASSERT (d <= 0x7FFFFFFE);
      HPX_ASSERT (e <= 0x7FFFFFFE);
      HPX_ASSERT (f <= 0x7FFFFFFE);
      return (std::uint_fast32_t)(((std::uint_fast64_t)a * b + (std::uint_fast64_t)c * d + (std::uint_fast64_t)e * f + sum) % 0x7FFFFFFF);
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
      return (std::uint_fast32_t)(((std::uint_fast64_t)a * b + (std::uint_fast64_t)c * d + (std::uint_fast64_t)e * f + (std::uint_fast64_t)g * h + sum) % 0x7FFFFFFF);
    }

    /* The two constants x and y are special cases because they are easier to
     * multiply by on 32-bit systems.  They are used as multipliers in the random
     * number generator.  The techniques for fast multiplication by these
     * particular values are in L'Ecuyer's papers; we don't use them yet. */

    inline std::uint_fast32_t mod_mul_x(std::uint_fast32_t a) {
      return mod_mul(a, 107374182);
    }

    inline std::uint_fast32_t mod_mul_y(std::uint_fast32_t a) {
      return mod_mul(a, 104480);
    }

    inline std::uint_fast32_t mod_mac_y(std::uint_fast32_t sum, std::uint_fast32_t a) {
      return mod_mac(sum, a, 104480);
    }
}}

#endif /* GRAPH500_MOD_ARITH_64BIT_HPP */
