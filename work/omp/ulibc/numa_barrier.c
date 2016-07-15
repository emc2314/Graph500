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
#include <math.h>
#include <assert.h>
#include <ulibc.h>
#include <common.h>

#if _POSIX_BARRIERS > 0
static struct NUMA_barrier_t {
  int lnp;
  pthread_barrier_t barrier;
} *__barrier[MAX_NODES] = { NULL };
#endif // _POSIX_BARRIERS

int ULIBC_init_numa_barriers(void) {
  if (ULIBC_verbose())
    printf("ULIBC: enable NUMA-barrier using pthread_barrier_t (_POSIX_BARRIERS=%d)\n",
	   (int)_POSIX_BARRIERS);
#if _POSIX_BARRIERS > 0
  int wakeup_count = 0;
  
  if (ULIBC_verbose())
    printf("ULIBC_get_online_nodes(): %d nodes\n", ULIBC_get_online_nodes());
  
  for (int k = 0; k < ULIBC_get_online_nodes(); ++k) {
    struct NUMA_barrier_t *bar = NULL;
    if ( __barrier[k] ) {
      bar = (struct NUMA_barrier_t *)__barrier[k];
    } else {
      ++wakeup_count;
      const size_t size = ROUNDUP(sizeof(struct NUMA_barrier_t), ULIBC_align_size());
      bar = (struct NUMA_barrier_t *)NUMA_touched_malloc(size, k);
      __barrier[k] = bar;
    }
    assert( bar );
    bar->lnp = ULIBC_get_online_cores(k);
    pthread_barrier_init(&bar->barrier, NULL, bar->lnp);
  }
  
  if (ULIBC_verbose())
    printf("ULIBC: woke up %d NUMA-barriers\n", wakeup_count);
#else
  if (ULIBC_verbose())
    printf("ULIBC: Not found pthread_barrier\n");
#endif // _POSIX_BARRIERS
  return 0;
}


void ULIBC_node_barrier(void) {
  if ( omp_in_parallel() ) {
#if _POSIX_BARRIERS > 0
    const struct numainfo_t ni = ULIBC_get_current_numainfo();
    struct NUMA_barrier_t *barrier = __barrier[ni.node];
    assert( barrier );
    pthread_barrier_wait(&barrier->barrier);
#else
    OMP("omp barrier");
#endif // _POSIX_BARRIERS
  }
}
