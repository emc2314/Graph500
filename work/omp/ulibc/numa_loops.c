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
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <ulibc.h>
#include <common.h>

static int64_t *__counter[MAX_NODES] = { NULL };
static int64_t *__loopend[MAX_NODES] = { NULL };

int ULIBC_init_numa_loops(void) {
  size_t size[MAX_CPUS];
  void *pool[MAX_CPUS];
  for (int i = 0; i < ULIBC_get_online_nodes(); ++i) {
    size[i] = ROUNDUP(sizeof(int64_t), ULIBC_align_size());
    pool[i] = NUMA_malloc(size[i], i);
    __counter[i] = &((int64_t *)pool[i])[00];
    __loopend[i] = &((int64_t *)pool[i])[64];
  }
  ULIBC_touch_memories(size,pool);
  return 0;
}

void ULIBC_clear_numa_loop(int64_t loopstart, int64_t loopend) {
  int node = 0, core = 0;
  if ( omp_in_parallel() ) {
    const int id = omp_get_thread_num();
    node = ULIBC_get_numainfo(id).node;
    core = ULIBC_get_numainfo(id).core;
  }
  if (core == 0) {
    *__counter[node] = loopstart;
    *__loopend[node] = loopend;
  }
}

int ULIBC_numa_loop(int64_t chunk, int64_t *start, int64_t *end) {
  int node = 0;
  if ( omp_in_parallel() ) {
    node = ULIBC_get_numainfo( omp_get_thread_num() ).node;
  }
  const int64_t t = add_and_fetch_int64(__counter[node], chunk);
  const int64_t term = *__loopend[node];
  if (t - chunk > term) {
    return 1;
  } else {
    *start = t - chunk;
    *end = t < term ? t : term;  
    return 0;
  }
}
