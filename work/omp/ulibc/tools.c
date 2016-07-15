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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>
#include <ulibc.h>

#include <common.h>

double get_msecs(void) {
  struct timeval tv;
  assert( !gettimeofday(&tv, NULL) );
  return (double)tv.tv_sec*1e3 + (double)tv.tv_usec*1e-3;
}

unsigned long long get_usecs(void) {
  struct timeval tv;
  assert( !gettimeofday(&tv, NULL) );
  return (unsigned long long)tv.tv_sec*1000000 + tv.tv_usec;
}

long long getenvi(char *env, long long def) {
  if (env && getenv(env)) {
    return atoll(getenv(env));
  } else {
    return def;
  }
}

double getenvf(char *env, double def) {
  if (env && getenv(env)) {
    return atof(getenv(env));
  } else {
    return def;
  }
}

size_t uniq(void *base, size_t nmemb, size_t size,
            void (*sort)(void *base, size_t nmemb, size_t size,
                         int (*compar)(const void *, const void *)),
            int (*compar)(const void *, const void *)) {
  size_t i, crr;
  sort(base, nmemb, size, compar);
  for (i = crr = 1; i < nmemb; i++) {
    if ( compar(base+(i-1)*size, base+i*size) != 0 ) {
      if (base+crr*size != base+i*size ) 
        memcpy(base+crr*size, base+i*size, size);
      crr++;
    }
  }
  return crr;
}

#define PARENT(x) ( ((x)-1) >> 1 )
#define LEFT(x)   ( ((x) << 1) + 1 )

static inline void heapify(void *base, size_t size, size_t i,
			   int (*compar)(const void *, const void *)) {
  void *crr = memcpy(alloca(size), base+i*size, size);
  while ( i > 0 && compar(base+PARENT(i)*size, crr) < 0 ) {
    memcpy(base+i*size, base+PARENT(i)*size, size);
    i = PARENT(i);
  }
  memcpy(base+i*size, crr, size);
}

static inline void extractmin(void *base, size_t size, size_t hqsz, void *buf,
			      int (*compar)(const void *, const void *)) {
  size_t i = 0, left = 1;
  void *HQ_hqsz = memcpy(buf, base+hqsz*size, size);
  
  /* left and right */
  size_t next;
  while ( left+1 < hqsz ) {
    if ( compar(base+left*size, base+(left+1)*size) > 0 ) {
      next = left;
    } else {
      next = left+1;
    }
    if ( compar(base+next*size, HQ_hqsz) > 0 ) {
      memcpy(base+i*size, base+next*size, size);
      i = next;
      left = LEFT(i);
    } else {
      break;
    }
  }
  /* left only */
  if ( left+1 == hqsz && compar(base+left*size, HQ_hqsz) > 0 ) {
    memcpy(base+i*size, base+left*size, size);
    i = left;
  }
  if (i != hqsz) {
    memcpy(base+i*size, HQ_hqsz, size);
  }
}

void uheapsort(void *base, size_t nmemb, size_t size,
               int (*compar)(const void *, const void *)) {
  /* heapify */
  for (size_t i = 0; i < nmemb; ++i) {
    heapify(base, size, i, compar);
  }
  
  size_t bufsz = ROUNDUP( size, ULIBC_align_size() );
  void *scratch = malloc(bufsz);
  void *buf = malloc(bufsz);
  memset(scratch, 0x00, bufsz);
  memset(buf, 0x00, bufsz);
  
  /* extractmin */
  for (size_t i = 0; i < nmemb; i++) {
    memcpy(scratch, base, size);
    extractmin(base, size, (nmemb-1)-i, buf, compar);
    memcpy(base+(nmemb-1-i)*size, scratch, size);
  }
  
  free( scratch );
  free( buf );
}

/* ------------------------- *
 * Local variables:          *
 * c-basic-offset: 2         *
 * indent-tabs-mode: nil     *
 * End:                      *
 * ------------------------- */
