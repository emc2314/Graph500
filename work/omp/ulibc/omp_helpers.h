/* ---------------------------------------------------------------------- *
 *
 * Copyright (C) 2014 Yuichiro Yasui < yuichiro.yasui@gmail.com >
 *
 * This file is part of ULIBC.
 *
 * ULIBC is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ULIBC is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with ULIBC.  If not, see <http://www.gnu.org/licenses/>.
 * ---------------------------------------------------------------------- */
#ifndef OMP_HELPER_H
#define OMP_HELPER_H

#if defined(_OPENMP)
#  define       OMP(x) _Pragma(x)
#  define  INITLOCK(x) omp_init_lock(x)
#  define   SETLOCK(x) omp_set_lock(x)
#  define UNSETLOCK(x) omp_unset_lock(x)
#  define  TESTLOCK(x) omp_test_lock(x)
#  define   DELLOCK(x) omp_destroy_lock(x)
#  include <omp.h>
#else
#  define       OMP(x)
#  define  INITLOCK(x)
#  define   SETLOCK(x)
#  define UNSETLOCK(x)
#  define  TESTLOCK(x) (1)
#  define   DELLOCK(x)
/* #if defined(__GNUC__) */
/* static int omp_get_thread_num (void) __attribute__((unused)); */
/* static int omp_get_num_threads(void) __attribute__((unused)); */
/* static int omp_get_max_threads(void) __attribute__((unused)); */
/* static int omp_in_parallel    (void) __attribute__((unused)); */
/* int omp_get_thread_num (void) { return 0; } */
/* int omp_get_num_threads(void) { return 1; } */
/* int omp_get_max_threads(void) { return 1; } */
/* int omp_in_parallel    (void) { return 0; } */
/* #else */
/* static int omp_get_thread_num(void) { return 0; } */
/* static int omp_get_num_threads(void) { return 1; } */
/* #endif */
#endif

#endif /* OMP_HELPER_H */
