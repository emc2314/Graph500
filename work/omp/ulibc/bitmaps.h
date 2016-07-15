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

#ifndef BITMAPS_H
#define BITMAPS_H

#ifndef bitmap_t
#define bitmap_t uint64_t
#endif
#ifndef UI64_SHIFT
#define UI64_SHIFT (64ULL)
#endif
#ifndef UI64_SHIFT2
#define UI64_SHIFT2 (6ULL)
#endif

#ifndef BITMAP_WRD
#define BITMAP_WRD(idx) ( (idx) >>  UI64_SHIFT2   )
#endif
#ifndef BITMAP_OFF
#define BITMAP_OFF(idx) ( (idx)  & (UI64_SHIFT-1) )
#endif
#ifndef SUMMAP_WRD
#define SUMMAP_WRD(idx) ( BITMAP_WRD( BITMAP_WRD(idx) ) )
#endif
#ifndef SUMMAP_OFF
#define SUMMAP_OFF(idx) ( BITMAP_OFF( BITMAP_WRD(idx) ) )
#endif
#ifndef INDEX
#define INDEX(idx,off)  ( ((bitmap_t)(idx) << UI64_SHIFT2) + off )
#endif
#ifndef BITMASK
#define BITMASK(a, b, c)    ((((bitmap_t)(a) >> (b)) & (c)))
#endif

static inline void SET_BITWRD(bitmap_t *wrd, uint64_t v) {
  *wrd |=  (1ULL << v);
}
static inline void UNSET_BITWRD(bitmap_t *wrd, uint64_t v) {
  *wrd &= ~(1ULL << v);
}
static inline int ISSET_BITWRD(bitmap_t *wrd, uint64_t v) {
  return ( *wrd & (1ULL << v) ) != 0;
}

static inline void SET_BITMAP(bitmap_t *map, uint64_t v) {
  map[ BITMAP_WRD(v) ] |=  (1ULL << BITMAP_OFF(v));
}
static inline void UNSET_BITMAP(bitmap_t *map, uint64_t v) {
  map[ BITMAP_WRD(v) ] &= ~(1ULL << BITMAP_OFF(v));
}
static inline int ISSET_BITMAP(bitmap_t *map, uint64_t v) {
  return ( map[ BITMAP_WRD(v) ] & (1ULL << BITMAP_OFF(v)) ) != 0;
}

static inline void SET_SUMMAP(bitmap_t *map, uint64_t v) {
  map[ SUMMAP_WRD(v) ] |=  (1ULL << SUMMAP_OFF(v));
}
static inline void UNSET_SUMMAP(bitmap_t *map, uint64_t v) {
  map[ SUMMAP_WRD(v) ] &= ~(1ULL << SUMMAP_OFF(v));
}
static inline int ISSET_SUMMAP(bitmap_t *map, uint64_t v) {
  return ( map[ SUMMAP_WRD(v) ] & (1ULL << SUMMAP_OFF(v)) ) != 0;
}

/*            __clang__               Clang/LLVM              */
/* [checked]  __GNUC__                gcc                     */
/*            __HP_cc,__HP_aCC        HP C/aC++               */
/* [checked]  __IBMC__,__IBMCPP__     IBM XL C/C++            */
/* [checked]  __ICC                   Intel ICC/ICPC          */
/*            _MSC_VER                Microsoft Visual Studio */
/*            __PGI                   Portland PGCC/PGCPP     */
/* [checked]  __SUNPRO_C,__SUNPRO_CC  Oracle Solaris Studio   */

/* Find-first-set (ffs) or Counting-trailing-zeros (ctz) */
/*     remark: ffs(x) = 1 + ctz(x) */
#if  defined(__SUNPRO_C) || defined(__SUNPRO_CC)
#  include <strings.h>
#  include <sys/atomic.h>
int ffsll(long long bits);
#endif
static inline int cnttz_uint64(bitmap_t bits) {
#if defined(__GNUC__)
  return __builtin_ctzll(bits);
#elif defined(__IBMC__) || defined(__IBMCPP__)
  return __cnttz8(bits);
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
  return ffsll(bits)-1;
#else
# error not found "ctzll"
#endif
}

static inline int popcount_uint64(bitmap_t bits) {
#if defined(__GNUC__)
  return __builtin_popcountll(bits);
#else
  return 0;
#endif
}

/* iterator */
static inline int next_bit_iter(bitmap_t *bits) {
  const int i = cnttz_uint64(*bits);
  UNSET_BITWRD(bits,i);
  return i;
}


/* fetch-and-{and,or} */
static inline int64_t fetch_and_add_int64(int64_t *p, int64_t incr) {
#if defined(__GNUC__)
  return __sync_fetch_and_add(p,incr);
#elif defined(__IBMC__) || defined(__IBMCPP__)
  return __fetch_and_addlp((volatile long *)p, (unsigned long)incr);
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
  return atomic_add_64_nv((uint64_t *)p,incr)-incr;
#else
  int64_t oldval;
#  if _OPENMP >= 200711
  OMP("omp atomic capture") {
    oldval = *p; *p += incr;
  }
#  else
  OMP("omp critical") {
    oldval = *p; *p += incr;
  }
  OMP("omp flush (p)");
#  endif
  return oldval;
#endif
}

static inline uint64_t fetch_and_add_uint64(uint64_t *p, uint64_t incr) {
#if defined(__GNUC__)
  return __sync_fetch_and_add(p,incr);
#elif defined(__IBMC__) || defined(__IBMCPP__)
  return __fetch_and_addlp((volatile long *)p, (unsigned long)incr);
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
  return atomic_add_64_nv((uint64_t *)p,incr)-incr;
#else
  uint64_t oldval;
#  if _OPENMP >= 200711
  OMP("omp atomic capture") {
    oldval = *p; *p += incr;
  }
#  else
  OMP("omp critical") {
    oldval = *p; *p += incr;
  }
  OMP("omp flush (p)");
#  endif
  return oldval;
#endif
}

static inline int64_t add_and_fetch_int64(int64_t *p, int64_t incr) {
#if defined(__GNUC__)
  return __sync_add_and_fetch(p,incr);
#elif defined(__IBMC__) || defined(__IBMCPP__)
  return __fetch_and_addlp((volatile long *)p, (unsigned long)incr) + incr;
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
  return atomic_add_64_nv((uint64_t *)p, incr);
#else
  int64_t oldval;
#  if _OPENMP >= 200711
  OMP("omp atomic capture") {
    oldval = *p; *p += incr;
  }
#  else
  OMP("omp critical") {
    oldval = *p; *p += incr;
  }
  OMP("omp flush (p)");
#  endif
  return oldval + incr;
#endif
}

static inline uint64_t fetch_and_or_uint64(uint64_t *p, uint64_t incr) {
#if defined(__GNUC__)
  return __sync_fetch_and_or(p,incr);
#elif defined(__IBMC__) || defined(__IBMCPP__)
  return __fetch_and_orlp((volatile unsigned long *)p, (unsigned long)incr);
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
  return atomic_or_64_nv((uint64_t *)p,incr)-incr;
#else
  int64_t oldval;
#  if _OPENMP >= 200711
  OMP("omp critical") {
    oldval = *p; *p |= incr;
  }
#  else
  OMP("omp atomic capture") {
    oldval = *p; *p |= incr;
  }
#  endif
  return oldval;
#endif
}


/* Test-and-set */
static inline uint64_t test_and_set_bitmap(bitmap_t *bitmaps, uint64_t x) {
#if defined(__SUNPRO_C) || defined(__SUNPRO_CC)
  return atomic_set_long_excl((ulong_t *)&bitmaps[BITMAP_WRD(x)], BITMAP_OFF(x));
#else
  return fetch_and_or_uint64((uint64_t *)&bitmaps[BITMAP_WRD(x)], (1ULL << BITMAP_OFF(x)));
#endif
}

static inline int is_test_and_set_bitmap(bitmap_t *bitmaps, uint64_t x) {
#if defined(__SUNPRO_C) || defined(__SUNPRO_CC)
  return atomic_set_long_excl((ulong_t *)&bitmaps[BITMAP_WRD(x)], BITMAP_OFF(x)) != 0;
#else
  return (test_and_set_bitmap(bitmaps,x) & (1ULL << BITMAP_OFF(x))) != 0;
#endif
}

#define IS_TEST_AND_SET_BITMAP(bitmaps,x) \
  (test_and_set_bitmap(bitmaps,x) & (1ULL << BITMAP_OFF(x)))

static inline uint64_t test_and_set_summary_bitmap(bitmap_t *bitmaps, uint64_t x) {
#if defined(__SUNPRO_C) || defined(__SUNPRO_CC)
  return atomic_set_long_excl((ulong_t *)&bitmaps[SUMMAP_WRD(x)], SUMMAP_OFF(x));
#else
  return fetch_and_or_uint64((uint64_t *)&bitmaps[SUMMAP_WRD(x)], (1ULL << SUMMAP_OFF(x)));
#endif
}

static inline int is_test_and_set_summary_bitmap(bitmap_t *bitmaps, uint64_t x) {
#if defined(__SUNPRO_C) || defined(__SUNPRO_CC)
  return atomic_set_long_excl((ulong_t *)&bitmaps[SUMMAP_WRD(x)], SUMMAP_OFF(x)) != 0;
#else
  return (test_and_set_bitmap(bitmaps,x) & (1ULL << SUMMAP_OFF(x))) != 0;
#endif
}

#endif /* BITMAPS_H */

