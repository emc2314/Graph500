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
#include <assert.h>
#include <sched.h>
#include <ulibc.h>
#include <common.h>

#if defined(__linux__) && !defined(__ANDROID__)
#include <syscall.h>
#include <sys/mman.h>
extern long int syscall(long int __sysno, ...);

static inline int mbind(void *addr, unsigned long len, int mode,
			unsigned long *nodemask, unsigned long maxnode, unsigned flags) {
  return syscall(SYS_mbind, addr, len, mode, nodemask, maxnode, flags);
}

#endif

/* #define ULIBC_USE_MMAP_MEMORY */

/* ------------------------------------------------------------
 * allocations
 * ------------------------------------------------------------ */
static void get_partial_range(long len, long off, long np, long id, long *ls, long *le) {
  const long qt = len / np;
  const long rm = len % np;
  *ls = qt * (id+0) + (id+0 < rm ? id+0 : rm) + off;
  *le = qt * (id+1) + (id+1 < rm ? id+1 : rm) + off;
}

char *NUMA_memory_name(void) {
#ifndef ULIBC_USE_MMAP_MEMORY
#  ifdef USE_MALLOC
  return "malloc";
#  else
  return "posix_memalign";
#  endif
#else
  return "mmap";
#endif  
}

void *NUMA_malloc(size_t size, const int onnode) {
  if (size == 0)
    return NULL;
  
  const int node = ULIBC_get_online_nodeidx(onnode);
  
  void *p = NULL;
#ifndef ULIBC_USE_MMAP_MEMORY
#  ifdef USE_MALLOC
#warning using malloc
  p = malloc(size);
  if (ULIBC_verbose() >= 2)
    printf("ULIBC: NUMA %d on socket %d allocates %ld Bytes using malloc\n", onnode, node, size);
#  else
  posix_memalign((void *)&p, ULIBC_align_size(), size);
  if (ULIBC_verbose() >= 2)
    printf("ULIBC: NUMA %d on socket %d allocates %ld Bytes using posix_memalign "
	   "(%ld bytes aligned)\n", onnode, node, size, ULIBC_align_size());
#  endif
#else
#  ifdef __ia64__
#    define ADDR (void *)(0x8000000000000000UL)
#    define FLAGS (MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED)
#  else
#    define ADDR (void *)(0x0UL)
#    define FLAGS (MAP_PRIVATE | MAP_ANONYMOUS)
#  endif
  p = mmap(ADDR, size, (PROT_READ | PROT_WRITE), FLAGS, 0, 0);
  if (ULIBC_verbose() >= 2)
    printf("ULIBC: NUMA %d on socket %d allocates %ld Bytes using mmap "
	   "(%ld bytes aligned)\n", onnode, node, size, ULIBC_align_size());
  if ( ULIBC_enable_numa_mapping() &&
       ( 0 <= onnode && onnode < ULIBC_get_online_nodes() ) ) {
    /* hwloc_set_area_membind_nodeset(ULIBC_get_hwloc_topology(), p, size, */
    /* 				   ULIBC_get_node_hwloc_obj(node)->nodeset, */
    /* 				   HWLOC_MEMBIND_DEFAULT, 0); */
#if defined(__linux__) && !defined(__ANDROID__)
#warning using mmap && mbind
    unsigned long mask[MAX_NODES/sizeof(unsigned long)] = {0};
    const int wrd = node / 64, off = node % 64;
    mask[wrd] = (1ULL << off);
    enum MBIND_MODE {
      MPOL_MF_STRICT   = (1<<0),	/* Verify existing pages in the mapping */
      MPOL_MF_MOVE     = (1<<1),	/* Move pages owned by this process to conform to mapping */
      MPOL_MF_MOVE_ALL = (1<<2) 	/* Move every page to conform to mapping */
    };
    enum MBIND_FLAGS {
      MPOL_DEFAULT     = (0),
      MPOL_PREFERRED   = (1),
      MPOL_BIND        = (2),
      MPOL_INTERLEAVE  = (3)
    };
    mbind(p, size, MPOL_PREFERRED, mask, ULIBC_get_num_nodes(), MPOL_PREFERRED);
#endif
  }
#endif

  return p;
}


void *NUMA_touched_malloc(size_t sz, int onnode) {
  void *p = NUMA_malloc(sz, onnode);
  OMP("omp parallel") {
    struct numainfo_t ni = ULIBC_get_current_numainfo();
    if ( onnode == ni.node ) {
      const long stride = ULIBC_page_size(ni.node);
      unsigned char *x = p;
      long ls, le;
      get_partial_range(sz, 0, ni.lnp, ni.core, &ls, &le);
      /* printf("[%d-%d] sz is %lld -> %lld,%lld\n", ni.node, ni.core, sz, ls, le); */
      for (long k = ls; k < le; k += stride) x[k] = -1;
    }
  }
  return p;
}

void ULIBC_touch_memories(size_t size[MAX_NODES], void *pool[MAX_NODES]) {
  if (!pool) {
    fprintf(stderr, "ULIBC: Wrong address: pool is %p\n", pool);
    return;
  }
  OMP("omp parallel") {
    struct numainfo_t ni = ULIBC_get_current_numainfo();
    unsigned char *addr = pool[ni.node];
    const size_t sz = size[ni.node];
    if (addr) {
      long ls, le;
      const long stride = ULIBC_page_size(ni.node);
      get_partial_range(sz, 0, ni.lnp, ni.core, &ls, &le);
      assert( ni.lnp == ULIBC_get_online_cores( ni.node ) );
      /* printf("%d-%d of #nodes:%d, #procs:%d [%ld:%ld]\n", */
      /* 	     ni.node, ni.core, */
      /* 	     get_numa_online_nodes(), get_numa_online_procs(), ls, le); */
      for (long k = ls; k < le; k += stride) addr[k] = ~0;
    }
  }
}

void NUMA_free(void *p) {
#ifndef ULIBC_USE_MMAP_MEMORY
  if (p) free(p);
#else
  (void)p;
#endif
}
