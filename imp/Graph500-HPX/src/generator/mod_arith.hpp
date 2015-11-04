/* Copyright (C) 2010 The Trustees of Indiana University.                  */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#ifndef GRAPH500_MOD_ARITH_HPP
#define GRAPH500_MOD_ARITH_HPP

#include <generator/user_settings.hpp>

/* Various modular arithmetic operations for modulus 2^31-1 (0x7FFFFFFF).
 * These may need to be tweaked to get acceptable performance on some platforms
 * (especially ones without conditional moves). */

/* This code is now just a dispatcher that chooses the right header file to use
 * per-platform. */

#ifdef FAST_64BIT_ARITHMETIC
#include <generator/mod_arith_64bit.hpp>
#else
#include <generator/mod_arith_32bit.hpp>
#endif

#endif /* MOD_ARITH_H */
