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
#ifndef ULIBC_H
#define ULIBC_H

#ifndef ULIBC_VERSION
#define ULIBC_VERSION "ULIBC (version 1.10 for Linux)"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <pthread.h>
#include <omp_helpers.h>
#include <bitmaps.h>

/* evironment values */
/*   ULIBC_ALIGNSIZE          ... alignment size in bytes */
/*   ULBIC_AVOID_HTCORE       ... avoids Hyperthteading (HT) cores */
/*   ULIBC_AFFINITY           ... sets affinity types (scatter or compact), and binding types (socket, core, phycore) */
/*   ULIBC_USE_SCHED_AFFINITY ... uses scheduler affinity */

#if defined (__cplusplus)
extern "C" {
#endif
  /* init.c */
  extern int ULIBC_init(void);
  extern int ULIBC_verbose(void);
  extern char *ULIBC_version(void);  
  
  /* topology.c */
  extern int ULIBC_init_topology(void);
  extern int ULIBC_get_num_procs(void);
  extern int ULIBC_get_num_nodes(void);
  extern int ULIBC_get_num_cores(void);
  extern int ULIBC_get_num_smts(void);
  extern size_t ULIBC_page_size(unsigned nodeidx);
  extern size_t ULIBC_memory_size(unsigned nodeidx);
  extern size_t ULIBC_total_memory_size(void);
  extern size_t ULIBC_align_size(void);
  struct cpuinfo_t {
    int id, node, core, smt;
    //hwloc_obj_t obj;
  };
  extern struct cpuinfo_t ULIBC_get_cpuinfo(unsigned procidx);
  extern void ULIBC_print_topology(FILE *fp);
  
  /* online_topology.c */
  extern int ULIBC_init_online_topology(void);
  extern int ULIBC_get_max_online_procs(void);
  extern int ULIBC_enable_online_procs(void);
  extern int ULIBC_get_online_procidx(unsigned idx);
  extern void ULIBC_print_online_topology(FILE *fp);
  
  /* numa_mapping.c */
  extern int ULIBC_init_numa_policy(void);
  extern int ULIBC_init_numa_mapping(void);
  
  extern int ULIBC_use_affinity(void);
  extern int ULIBC_enable_numa_mapping(void);
  extern int ULIBC_get_current_mapping(void);
  extern int ULIBC_get_current_binding(void);
  extern const char *ULIBC_get_current_mapping_name(void);
  extern const char *ULIBC_get_current_binding_name(void);
  
  extern int ULIBC_get_online_procs(void);
  extern int ULIBC_get_online_nodes(void);
  extern int ULIBC_get_online_cores(int node);
  extern int ULIBC_get_online_nodeidx(int node);
  
  extern int ULIBC_get_num_threads(void);
  extern void ULIBC_set_num_threads(int nt);
  extern int ULIBC_set_affinity_policy(int nt, int map, int bind);
  
  struct numainfo_t {
    int id;	/* Thread ID */
    int proc;	/* Processor ID for "cpuinfo" */
    int node;	/* NUMA node ID */
    int core;	/* NUMA local-core ID */
    int lnp;	/* Number of local cores in NUMA node */
  };
  extern struct numainfo_t ULIBC_get_numainfo(int tid);
  extern struct numainfo_t ULIBC_get_current_numainfo(void);
  extern void ULIBC_print_mapping(FILE *fp);
  
  enum affinity_type_t {
    NULL_AFFINITY,
    ULIBC_AFFINITY,
    SCHED_AFFINITY
  };
  enum map_policy_t {
    SCATTER_MAPPING = 0x00,
    COMPACT_MAPPING = 0x01,
  };
  enum bind_level_t {
    THREAD_TO_CORE          = 0x00,
    THREAD_TO_PHYSICAL_CORE = 0x01,
    THREAD_TO_SOCKET        = 0x02,
  };
  
  /* numa_threads.c */
  extern int ULIBC_init_numa_threads(void);
  extern int ULIBC_bind_thread(void);
  extern int ULIBC_unbind_thread(void);
  extern int ULIBC_is_bind_thread(int pos, int target);
  extern long ULIBC_get_num_bind_threads(int id);
      
  /* numa_barrier.c */
  extern int ULIBC_init_numa_barriers(void);
  extern void ULIBC_node_barrier(void);  
  
  /* numa_malloc.c */
  extern char *NUMA_memory_name(void);
  extern void *NUMA_malloc(size_t size, const int onnode);
  extern void *NUMA_touched_malloc(size_t sz, int onnode);
  extern void ULIBC_touch_memories(size_t size[], void *pool[]);
  extern void NUMA_free(void *p);
  
  /* numa_loops.c */
  extern int ULIBC_init_numa_loops(void);
  extern void ULIBC_clear_numa_loop(int64_t loopstart, int64_t loopend);
  extern int ULIBC_numa_loop(int64_t chunk, int64_t *start, int64_t *end);
  
  /* tools.c */
  extern double get_msecs(void);
  extern unsigned long long get_usecs(void);
  extern long long getenvi(char *env, long long def);
  extern double getenvf(char *env, double def);
  extern size_t uniq(void *base, size_t nmemb, size_t size,
		     void (*sort)(void *base, size_t nmemb, size_t size,
				  int(*compar)(const void *, const void *)),
		     int (*compar)(const void *, const void *));
  extern void uheapsort(void *base, size_t nmemb, size_t size,
			int (*compar)(const void *, const void *));
#if defined (__cplusplus)
}
#endif
#endif /* ULIBC_H */
