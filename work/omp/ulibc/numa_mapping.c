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
#include <ulibc.h>
#include <common.h>

/* affinity policy */
int __avoid_htcore = 0;
int __use_affinity = NULL_AFFINITY;
int __mapping_policy = SCATTER_MAPPING;
int __binding_policy = THREAD_TO_CORE;

/* affinity */
int __online_procs;
int __online_nodes;
int __online_cores[MAX_NODES];
int __node_mapping[MAX_NODES];
struct numainfo_t __numainfo[MAX_CPUS];

static void get_sorted_procs(int *sorted_proc);
static int make_numainfo(int *sorted_proc);

/* ------------------------------------------------------------
 * Init. functions
 * ------------------------------------------------------------ */
int ULIBC_init_numa_policy(void) {
  __avoid_htcore = getenvi("ULBIC_AVOID_HTCORE", 0);
  if (ULIBC_verbose())
    printf("ULIBC: ULBIC_AVOID_HTCORE is %d\n", __avoid_htcore);

  /* default policy */
  const char *affinity_env = getenv("ULIBC_AFFINITY");
  if ( affinity_env ) {
    if (ULIBC_verbose())
      printf("ULIBC: ULIBC_AFFINITY is %s\n", affinity_env);
    char buff[256];
    strcpy(buff, affinity_env);
    char *affi_name = strtok(buff, ":");
    char *bind_name = strtok(NULL, ":");
    if ( affi_name ) {
      if      ( !strcmp(affi_name, "scatter") ) __mapping_policy = SCATTER_MAPPING;
      else if ( !strcmp(affi_name, "compact") ) __mapping_policy = COMPACT_MAPPING;
      else {
	printf("Unkrown affinity policy '%s'.\n"
	       "  ULIBC supports 'scatter' or 'compact'.\n", affi_name);
	exit(1);
      }
    }
    if ( bind_name ) {
      if      ( !strcmp(bind_name, "core")     ) __binding_policy = THREAD_TO_CORE;
      else if ( !strcmp(bind_name, "physcore") ) __binding_policy = THREAD_TO_PHYSICAL_CORE;
      else if ( !strcmp(bind_name, "socket")   ) __binding_policy = THREAD_TO_SOCKET;
      else {
	printf("Unkrown binding policy '%s'.\n"
	       "  ULIBC supports as 'core', 'physcore', or 'socket'.\n", bind_name);
	exit(1);
      }
    }
  }
  return 0;
}

int ULIBC_init_numa_mapping(void) {
  /* affinity */
  int sorted_proc[MAX_CPUS];
  TIMED( get_sorted_procs(sorted_proc) );
  
  if ( ULIBC_verbose() >= 3 ) {
    for (int i = 0; i < ULIBC_get_max_online_procs(); ++i) {
      const int idx = sorted_proc[i];
      struct cpuinfo_t ci = ULIBC_get_cpuinfo(idx);
      printf("ULIBC: CPU[%03d] Processor: %3d, Package: %2d, Core: %2d, SMT: %2d\n",
	     idx, ci.id, ci.node, ci.core, ci.smt);
    }
  }
  if ( ULIBC_verbose() ) { 
    switch (__use_affinity) {
    case NULL_AFFINITY:
      printf("ULIBC: w/o affinity\n");
      break;
    case ULIBC_AFFINITY:
      printf("ULIBC: using ULIBC affinity (%s:%s)\n",
	     ULIBC_get_current_mapping_name(), ULIBC_get_current_binding_name());
      break;
    case SCHED_AFFINITY:
      if (getenv("KMP_AFFINITY")) {
	printf("ULIBC: using Intel affinity (%s)\n", getenv("KMP_AFFINITY"));
      } else {
	printf("ULIBC: using scheduler affinity\n");
      }
      break;
    }
  }
  
  /* threads */
  __online_procs = getenvi("OMP_NUM_THREADS", ULIBC_get_max_online_procs());
  __online_procs = MIN(__online_procs, ULIBC_get_max_online_procs());
  if ( ULIBC_verbose() ) printf("ULIBC: #online cpus  is %d\n", __online_procs);
  
  omp_set_num_threads(__online_procs);
  if ( ULIBC_verbose() ) printf("ULIBC: omp_set_num_threads=%d\n", __online_procs);
  
  /* NUMA info */
  TIMED( __online_nodes = make_numainfo(sorted_proc) );
  
  if ( ULIBC_verbose() >= 2 )
    ULIBC_print_mapping(stdout);
  
  return 0;
}


/* ------------------------------------------------------------
 * Set/Get functions
 * ------------------------------------------------------------ */
int ULIBC_use_affinity(void) { return __use_affinity; }
int ULIBC_enable_numa_mapping(void) {
  switch ( ULIBC_use_affinity() ) {
  case NULL_AFFINITY: return 0;
  case ULIBC_AFFINITY:
  case SCHED_AFFINITY: return 1;
  default: return 0;
  }
}

int ULIBC_get_current_mapping(void) { return __mapping_policy; }
int ULIBC_get_current_binding(void) { return __binding_policy; }
const char *ULIBC_get_current_mapping_name(void) {
  if ( ULIBC_enable_numa_mapping() ) {
    switch ( ULIBC_get_current_mapping() ) {
    case SCATTER_MAPPING: return "scatter";
    case COMPACT_MAPPING: return "compact";
    default:              return "unknown";
    }
  } else {
    return "disable";
  }
}
const char *ULIBC_get_current_binding_name(void) {
  if ( ULIBC_enable_numa_mapping() ) {
    switch ( ULIBC_get_current_binding() ) {
    case THREAD_TO_CORE:          return "core";
    case THREAD_TO_PHYSICAL_CORE: return "physcore";
    case THREAD_TO_SOCKET:        return "socket";
    default:                      return "unknown";
    }
  } else {
    return "disable";
  }
}

int ULIBC_get_online_procs(void) { return __online_procs; }
int ULIBC_get_online_nodes(void) { return __online_nodes; }
int ULIBC_get_online_cores(int node) { return __online_cores[node]; }
int ULIBC_get_online_nodeidx(int node) { return __node_mapping[node]; }

int ULIBC_get_num_threads(void) {
  return __online_procs;
}
void ULIBC_set_num_threads(int nt) {
  omp_set_num_threads(nt);
  ULIBC_set_affinity_policy(nt, ULIBC_get_current_mapping(), ULIBC_get_current_binding());
}
int ULIBC_set_affinity_policy(int nt, int map, int bind) {
  /* set */
  omp_set_num_threads(nt);
  __mapping_policy = map;
  __binding_policy = bind;
  
  /* init. */
  ULIBC_init_numa_mapping();
  ULIBC_init_numa_threads();
  ULIBC_init_numa_barriers();
  ULIBC_init_numa_loops();
  return 0;
}


/* get numa info */
struct numainfo_t ULIBC_get_numainfo(int tid) {
  if ( !ULIBC_enable_numa_mapping() ||
       (tid < 0 || ULIBC_get_max_online_procs() <= tid) ) {
    return (struct numainfo_t){
      .id = tid, .proc = tid, .node = 0, .core = tid,
	.lnp = ULIBC_get_max_online_procs() };
  } else {
    return __numainfo[tid];
  }
}

/* get numa info of current thread */
struct numainfo_t ULIBC_get_current_numainfo(void) {
  struct numainfo_t ni = ULIBC_get_numainfo( omp_get_thread_num() );
  ULIBC_bind_thread();
  return ni;
}

/* print numa mapping */
void ULIBC_print_mapping(FILE *fp) {
  if (!fp) return;
  for (int j = 0; j < ULIBC_get_online_nodes(); ++j) {
    int node = ULIBC_get_online_nodeidx(j);
    printf("ULIBC: [NUMA node %03d on socket%03d] (#cores: %3d) = {\n",
	   j, node, ULIBC_get_online_cores(j));
    printf("ULIBC:     Threads    = [");
    for (int i = 0; i < ULIBC_get_online_procs(); ++i) {
      struct numainfo_t ni = ULIBC_get_numainfo(i);
      if ( ULIBC_get_cpuinfo(ni.proc).node == node ) {
	printf("%3d, ", i);
      }
    }
    printf("]\n");
    printf("ULIBC:     Processors = [");
    for (int i = 0; i < ULIBC_get_online_procs(); ++i) {
      struct numainfo_t ni = ULIBC_get_numainfo(i);
      if (ULIBC_get_cpuinfo(ni.proc).node == node) {
	printf("%3d, ", ULIBC_get_cpuinfo(ni.proc).id);
      }
    }
    printf("]\n");
    printf("ULIBC: }\n");
  }
}


/* get processor list with CPU affinity */
static int cmpr_scatter(const void *a, const void *b);
static int cmpr_compact(const void *a, const void *b);
static int cmpr_compact_avoid_ht(const void *a, const void *b);
static void get_scheduler_affinity(int *sorted_proc);

static void get_sorted_procs(int *sorted_proc) {
  /* fill sorted_list with online processors */
  for (int i = 0; i < ULIBC_get_max_online_procs(); ++i) {
    sorted_proc[i] = ULIBC_get_online_procidx(i);
  }
  if ( !ULIBC_enable_online_procs() ) {
    if (ULIBC_verbose())
      printf("ULIBC: ULIBC_enable_online_procs(): False\n");
    __use_affinity = NULL_AFFINITY;
    return ;
  } else {
    if (ULIBC_verbose())
      printf("ULIBC: ULIBC_enable_online_procs(): True\n");
  }
  
  if ( ULIBC_verbose() >= 2 ) {
    printf("ULIBC: Before: ");
    for (int i = 0; i < ULIBC_get_max_online_procs(); ++i)
      printf("%d ", ULIBC_get_cpuinfo( sorted_proc[i] ).id);
    printf("\n");
  }
  
  /* detecting or sorting */
  if ( getenvi("ULIBC_USE_SCHED_AFFINITY",0) ||
       getenv("KMP_AFFINITY") ) {
    __use_affinity = SCHED_AFFINITY;
    TIMED( get_scheduler_affinity(sorted_proc) );
  } else {
    __use_affinity = ULIBC_AFFINITY;
    switch ( __mapping_policy ) {
    case SCATTER_MAPPING:
      qsort(sorted_proc, ULIBC_get_max_online_procs(), sizeof(int), cmpr_scatter);
      break;
    case COMPACT_MAPPING:
      qsort(sorted_proc, ULIBC_get_max_online_procs(), sizeof(int),
	    (__avoid_htcore) ? cmpr_compact_avoid_ht : cmpr_compact);
      break;
    }
  }
  if ( ULIBC_verbose() >= 2 ) {
    printf("ULIBC: After: ");
    for (int i = 0; i < ULIBC_get_max_online_procs(); ++i)
      printf("%d ", ULIBC_get_cpuinfo( sorted_proc[i] ).id);
    printf("\n");
  }
}

static void get_scheduler_affinity(int *sorted_proc) {
#if defined(__linux__) && !defined(__ANDROID__)
  int dc = 0;
  OMP("omp parallel reduction(+:dc) num_threads(ULIBC_get_max_online_procs())") {
    int proc = -1, count = 0;
    const int id = omp_get_thread_num();
    
    cpu_set_t set;
    sched_getaffinity((pid_t)0, sizeof(cpu_set_t), &set);
    for (int i = 0; i < ULIBC_get_num_procs(); ++i) {
      if ( CPU_ISSET(i, &set) )
	++count, proc = i;
    }   
    
    /* dup. */
    if (count == 1) {
      sorted_proc[id] = proc;
    } else {
      ++dc;
    }
  }
  if (dc != 0) {
    printf("ULIBC: [error] detects %d duplicated threads\n"
	   "ULIBC: if you use the Intel-Thread-Affinity-Interface,\n"
	   "          rerun with \"KMP_AFFINITY=compact,granularity=fine\"\n"
	   "          rerun with \"KMP_AFFINITY=scatter,granularity=fine\"\n", dc);
    exit(1);
  }
  fflush(stdout);
#else
  (void)sorted_proc;
#endif
}

/* Comparison Functions of Processor ordering */
static int cmpr_proc(const void *a, const void *b);
static int cmpr_core(const void *a, const void *b);
static int cmpr_node(const void *a, const void *b);
static int cmpr_smt (const void *a, const void *b);

static int cmpr_scatter(const void *a, const void *b) {
  int ret;
  if ( (ret = cmpr_smt (a,b)) != 0 ) return ret;
  if ( (ret = cmpr_core(a,b)) != 0 ) return ret;
  if ( (ret = cmpr_node(a,b)) != 0 ) return ret;
  if ( (ret = cmpr_proc(a,b)) != 0 ) return ret;
  return ret;
}

static int cmpr_compact_avoid_ht(const void *a, const void *b) {
  int ret = 0;
  if ( (ret = cmpr_smt (a,b)) != 0 ) return ret;
  if ( (ret = cmpr_node(a,b)) != 0 ) return ret;
  if ( (ret = cmpr_core(a,b)) != 0 ) return ret;
  if ( (ret = cmpr_proc(a,b)) != 0 ) return ret;
  return ret;
}
static int cmpr_compact(const void *a, const void *b) {
  int ret = 0;
  if ( (ret = cmpr_node(a,b)) != 0 ) return ret;
  if ( (ret = cmpr_smt (a,b)) != 0 ) return ret;
  if ( (ret = cmpr_core(a,b)) != 0 ) return ret;
  if ( (ret = cmpr_proc(a,b)) != 0 ) return ret;
  return ret;
}

static int cmpr_proc(const void *a, const void *b) {
  const int _proc_a = ULIBC_get_cpuinfo(*(int *)a).id;
  const int _proc_b = ULIBC_get_cpuinfo(*(int *)b).id;
  if ( _proc_a < _proc_b) return -1;
  if ( _proc_a > _proc_b) return  1;
  return 0;
}

static int cmpr_core(const void *a, const void *b) {
  const int _core_a = ULIBC_get_cpuinfo(*(int *)a).core;
  const int _core_b = ULIBC_get_cpuinfo(*(int *)b).core;
  if ( _core_a < _core_b) return -1;
  if ( _core_a > _core_b) return  1;
  return 0;
}

static int cmpr_node(const void *a, const void *b) {
  const int _node_a = ULIBC_get_cpuinfo(*(int *)a).node;
  const int _node_b = ULIBC_get_cpuinfo(*(int *)b).node;
  if ( _node_a < _node_b) return -1;
  if ( _node_a > _node_b) return  1;
  return 0;
}

static int cmpr_smt(const void *a, const void *b) {
  const int _smt_a = ULIBC_get_cpuinfo(*(int *)a).smt;
  const int _smt_b = ULIBC_get_cpuinfo(*(int *)b).smt;
  if ( _smt_a < _smt_b) return -1;
  if ( _smt_a > _smt_b) return  1;
  return 0;
}

static int make_numainfo(int *sorted_proc) {
  int onnodes = 0;
  
  /* detect max. online_nodes/online_cores */
  bitmap_t online[MAX_NODES/64] = {0};
  for (int i = 0; i < ULIBC_get_online_procs(); ++i) {
    struct cpuinfo_t ci = ULIBC_get_cpuinfo( sorted_proc[i] );
    if ( !ISSET_BITMAP(online, ci.node) ) {
      if ( ULIBC_verbose() >= 2 )
      	printf("ULIBC: found Node %d\n", ci.node);
      SET_BITMAP(online, ci.node);
      __node_mapping[onnodes++] = ci.node;
    }
  }
  if ( ULIBC_verbose() >= 2 ) {
    for (int i = 0; i < onnodes; ++i) {
      printf("ULIBC: %d NUMA : %d\n", i, __node_mapping[i]);
    }
  }
    
  /* construct numainfo and online_cores */
  for (int i = 0; i < onnodes; ++i) {
    __online_cores[i] = 0;
  }
  for (int i = 0; i < ULIBC_get_online_procs(); ++i) {
    int node = 0;
    int cpu = sorted_proc[i];
    struct cpuinfo_t ci = ULIBC_get_cpuinfo(cpu);
    for (int j = 0; j < onnodes; ++j) {
      if (__node_mapping[j] == ci.node) node = j;
    }
    /* printf("%d (id:%d, cpu:%d, node:%d, core:%d)\n", i, i, cpu, node, __online_cores[node]); */
    __numainfo[i].id   = i;	/* thread index */
    __numainfo[i].proc = cpu;	/* serial index for cpuinfo */
    __numainfo[i].node = node;	/* NUMA node index  */
    __numainfo[i].core = __online_cores[node]++;
  }
  for (int i = 0; i < ULIBC_get_online_procs(); ++i) {
    __numainfo[i].lnp = __online_cores[ __numainfo[i].node ];
  }
  return onnodes;
}




