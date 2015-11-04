/* Copyright (C) 2009-2010 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#ifndef GRAPH500_GENERATOR_USER_SETTINGS_HPP
#define GRAPH500_GENERATOR_USER_SETTINGS_HPP

/* Settings for user modification ----------------------------------- */

/* #define GENERATOR_USE_PACKED_EDGE_TYPE -- 48 bits per edge */
#undef GENERATOR_USE_PACKED_EDGE_TYPE /* 64 bits per edge */

#define FAST_64BIT_ARITHMETIC /* Use 64-bit arithmetic when possible. */
/* #undef FAST_64BIT_ARITHMETIC -- Assume 64-bit arithmetic is slower than 32-bit. */

/* End of user settings ----------------------------------- */

#endif /* GRAPH500_GENERATOR_USER_SETTINGS_HPP  */
