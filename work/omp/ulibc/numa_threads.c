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
#include <string.h>
#include <ulibc.h>
#include <common.h>

#if defined(__linux__) && !defined(__ANDROID__)
#include <sched.h>
static cpu_set_t __default_cpuset[MAX_CPUS];
static cpu_set_t __bind_cpuset[MAX_CPUS];
#endif

static long __num_bind[MAX_CPUS] = {0};
static void bind_thread(void);

int ULIBC_init_numa_threads(void) {
  if ( ULIBC_use_affinity() == NULL_AFFINITY ) return 1;

  /* get default affinity */
#if defined(__linux__) && !defined(__ANDROID__) 
  OMP("omp parallel") {
    const int id = omp_get_thread_num();
    CPU_ZERO(&__default_cpuset[id]);
    CPU_ZERO(&__bind_cpuset[id]);
    assert( !sched_getaffinity((pid_t)0, sizeof(cpu_set_t), &__default_cpuset[id]) );
    bind_thread();
    assert( !sched_getaffinity((pid_t)0, sizeof(cpu_set_t), &__bind_cpuset[id]) );
  }
#endif
  
  if ( ULIBC_verbose() > 2 ) {
    const int cores_per_node = ULIBC_get_num_procs() / ULIBC_get_num_nodes();
    for (int i = 0; i < ULIBC_get_online_procs(); ++i) {
      struct numainfo_t ni = ULIBC_get_numainfo(i);
      struct cpuinfo_t ci = ULIBC_get_cpuinfo( ni.proc );
      printf("ULIBC: Thread: %3d of %d, NUMA: %2d-%02d (core=%2d), "
	     "Proc: %2d, Pkg: %2d, Core: %2d, Smt: %2d\t",
	     ni.id, ULIBC_get_online_procs(), ni.node, ni.core, ni.lnp,
	     ci.id, ci.node, ci.core, ci.smt);
      for (int i = 0; i < ULIBC_get_num_nodes(); ++i) {
	printf(ci.node == i ? "x" : "-");
      }
      printf("\t");
      for (int cpu = 0; cpu < ULIBC_get_num_procs(); ++cpu) {
	printf( ULIBC_is_bind_thread( ni.id, cpu ) ? "X" : "-" );
	if ((cpu+1) % cores_per_node == 0) printf(" ");
      }
      printf("\n");
    }
  }
  
  return 0;
}

static void bind_thread(void) {
  if ( ULIBC_use_affinity() != ULIBC_AFFINITY ) return;
  if ( !omp_in_parallel() ) return;
  
  /* constructs cpuset */
#if defined(__linux__) && !defined(__ANDROID__) 
  const int id = omp_get_thread_num();
  struct numainfo_t ni = ULIBC_get_numainfo(id);
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(ni.proc, &cpuset);
  
  switch ( ULIBC_get_current_binding() ) {
  case THREAD_TO_CORE: break;
    
  case THREAD_TO_PHYSICAL_CORE: {
    struct cpuinfo_t ci = ULIBC_get_cpuinfo( ni.proc );
    for (int u = 0; u < ULIBC_get_online_procs(); ++u) {
      struct cpuinfo_t cj = ULIBC_get_cpuinfo( ULIBC_get_numainfo(u).proc );
      if ( ci.node == cj.node && ci.core == cj.core )
  	CPU_SET(cj.id, &cpuset);
    }
    break;
  }
  case THREAD_TO_SOCKET: {
    struct cpuinfo_t ci = ULIBC_get_cpuinfo( ni.proc );
    for (int u = 0; u < ULIBC_get_online_procs(); ++u) {
      struct cpuinfo_t cj = ULIBC_get_cpuinfo( ULIBC_get_numainfo(u).proc );
      if ( ci.node == cj.node )
  	CPU_SET(cj.id, &cpuset);
    }
    break;
  }
  default: break;
  }
  
  /* binds */
  sched_setaffinity((pid_t)0, sizeof(cpu_set_t), &cpuset);
  sched_getaffinity((pid_t)0, sizeof(cpu_set_t), &__bind_cpuset[id]);
  ++__num_bind[id];
#endif
}

int ULIBC_bind_thread(void) {
  if ( ULIBC_use_affinity() != ULIBC_AFFINITY ) return 0;
  if ( !omp_in_parallel() ) return 0;
  
#if defined(__linux__) && !defined(__ANDROID__)
  const int id = omp_get_thread_num();
  cpu_set_t set;
  CPU_ZERO(&set);
  assert( !sched_getaffinity((pid_t)0, sizeof(cpu_set_t), &set) );
  if ( CPU_EQUAL(&set, &__bind_cpuset[id]) ) {
    return 0;
  } else {
    bind_thread();
    return 1;
  }
#endif
  return 0;
}

int ULIBC_unbind_thread(void) {
  if ( ULIBC_use_affinity() != ULIBC_AFFINITY ) return 0;
  if ( !omp_in_parallel() ) return 0;
  
#if defined(__linux__) && !defined(__ANDROID__)
  const int id = omp_get_thread_num();
  sched_setaffinity((pid_t)0, sizeof(cpu_set_t), &__default_cpuset[id]);
#endif
  
  return 1;
}

int ULIBC_is_bind_thread(int tid, int procid) {
#if defined(__linux__) && !defined(__ANDROID__)
  return CPU_ISSET(procid, &__bind_cpuset[tid]);
#else
  return 0;
#endif
}

long ULIBC_get_num_bind_threads(int id) {
  return __num_bind[id];
}
