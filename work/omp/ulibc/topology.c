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
#include <assert.h>
#include <string.h>
#include <ctype.h>
#include <dirent.h>
#include <ulibc.h>
#include <common.h>

#ifndef DEFAULT_PAGESIZE
#define DEFAULT_PAGESIZE (2 * (1UL << 20))
#endif

static size_t __pagesize[MAX_NODES] = { DEFAULT_PAGESIZE };
static size_t __memorysize[MAX_NODES] = {0};
static size_t __alignsize = 0;
static int __cpuinfo_count = 0;
static int __num_procs;
static int __num_nodes;
static int __num_cores;
static int __num_smts;
static struct cpuinfo_t __cpuinfo[MAX_CPUS] = { {0,0,0,0} };

#if defined(__linux__) && !defined(__ANDROID__)
#include <sys/sysinfo.h>
#define number_of_procs() get_nprocs()
#else
#include <omp.h>
#define number_of_procs() omp_get_num_procs()
#endif

static int fill_cpuinfo(struct cpuinfo_t *cpuinfo);
static int get_max_nodes(int ncpus, struct cpuinfo_t *cpuinfo);
static int get_uniq_cores(int ncpus, struct cpuinfo_t *cpuinfo, int cores[MAX_CPUS]);
static int make_smtid(int ncpus, struct cpuinfo_t *cpuinfo, 
		      int nnodes, int ncores, int uniq_cores[MAX_CPUS]);

int ULIBC_init_topology(void) {
  double t;
  
  /* get topology using HWLOC */
  if ( ULIBC_verbose() )
    printf("ULIBC: HWLOC API version: not used\n");
  
  /* Init. cpuinfo array */
  __num_procs = number_of_procs();
  __num_nodes = 1;
  __num_cores = __num_procs;
  __num_smts  = __num_procs;
  for (int i = 0; i < __num_procs; ++i) {
    __cpuinfo[i] = (struct cpuinfo_t){ .id = i, .node = 0, .core = i, .smt = 0 };
  }
  
  /* read cpuinfos from device file such as /sys/devices/system/{cpu,node}/.. */
  PROFILED( t, __cpuinfo_count = fill_cpuinfo(__cpuinfo) );
  
  if ( ULIBC_verbose() )
    printf("ULIBC: #cpuinfo is %d (%d cores)\n", __cpuinfo_count, __num_procs);
  if ( !(__num_procs == __cpuinfo_count) ) return 1;
  
  /* detect #nodes */
  PROFILED( t, __num_nodes = get_max_nodes(__num_procs, __cpuinfo) );
  if ( ULIBC_verbose() ) printf("ULIBC: #nodes is %d\n", __num_nodes);
  
  /* detect #cores */
  int uniq_cores[MAX_CPUS];
  PROFILED( t, __num_cores = get_uniq_cores(__num_procs, __cpuinfo, uniq_cores) );
  __num_cores *= __num_nodes;
  if ( ULIBC_verbose() ) printf("ULIBC: #cores is %d\n", __num_cores);

  /* set SMT ID and detect #SMTs */
  PROFILED( t, __num_smts = make_smtid(__num_procs, __cpuinfo, __num_nodes, __num_cores, uniq_cores) );
  __num_smts *= __num_cores;
  if ( ULIBC_verbose() ) printf("ULIBC: #smt is %d\n", __num_smts);
  
  
  if ( ULIBC_verbose() ) {
    if ( __num_procs != __cpuinfo_count ) {
      printf("ULIBC: cannot cpuinfos\n");
      printf("ULIBC: # CPUs is %d, # CPUinfos is %d\n", __num_procs, __cpuinfo_count);
      exit(1);
    }
  }
  __alignsize = getenvi("ULIBC_ALIGNSIZE", ULIBC_page_size(0));
  
  /* Check minimum node index  */
  int min_node = MAX_NODES;
  for (int i = 0; i < __cpuinfo_count; ++i) {
    min_node = MIN(min_node, __cpuinfo[i].node);
  }
  if ( ULIBC_verbose() )
    printf("ULIBC: min. node index is %d\n", min_node);
  assert( min_node >= 0 );
  
  if (__num_nodes == 0) __num_nodes = 1;
  if (__alignsize == 0) __alignsize = DEFAULT_PAGESIZE;
  for (int i = 0; i < __num_nodes; ++i) {
    if (__pagesize[i] == 0)
      __pagesize[i] = DEFAULT_PAGESIZE;
    __memorysize[i] = 0;
  }
  
  if ( ULIBC_verbose() ) {
    printf("ULIBC: # of Processor cores is %4d\n", ULIBC_get_num_procs());
    printf("ULIBC: # of NUMA nodes      is %4d\n", ULIBC_get_num_nodes());
    printf("ULIBC: # of Cores           is %4d\n", ULIBC_get_num_cores());
    printf("ULIBC: # of SMTs            is %4d\n", ULIBC_get_num_smts());
    printf("ULIBC: Alignment size       is %ld bytes\n", ULIBC_align_size());
  }
  
  if ( ULIBC_verbose() > 2 ) {
    ULIBC_print_topology(stdout);
  }
  return 0;
}



/* get number of processors, packages, cores/socket, and threads/core, page size, memory size */
int ULIBC_get_num_procs(void) { return __num_procs; }
int ULIBC_get_num_nodes(void) { return __num_nodes; }
int ULIBC_get_num_cores(void) { return __num_cores; }
int ULIBC_get_num_smts(void) { return __num_smts; }
size_t ULIBC_page_size(unsigned nodeidx) { return __pagesize[nodeidx]; }
size_t ULIBC_memory_size(unsigned nodeidx) { return __memorysize[nodeidx]; }
size_t ULIBC_align_size(void) { return __alignsize; }

size_t ULIBC_total_memory_size(void) {
  static size_t total = 0;
  if (total == 0) {
    for (int i = 0; i < ULIBC_get_num_nodes(); ++i)
      total += ULIBC_memory_size(i);
  }
  return total;
}

struct cpuinfo_t ULIBC_get_cpuinfo(unsigned procidx) {
  if (ULIBC_get_num_procs() <= (int)procidx)
    procidx %= ULIBC_get_num_procs();
  return __cpuinfo[procidx];
}

/* print cpuinfo */
void ULIBC_print_topology(FILE *fp) {
  if (!fp) return;
  for (int i = 0; i < ULIBC_get_num_nodes(); ++i) {
    fprintf(fp, "ULIBC: NUMA %d has %d CPU cores and %.3f GB (%ld bytes) memory (using %ld bytes page size)\n",
	    i,
	    ULIBC_get_num_procs() / ULIBC_get_num_nodes(),
	    1.0*ULIBC_memory_size(i)/(1UL<<30),
	    ULIBC_memory_size(i),
	    ULIBC_page_size(i));
  }
  for (int i = 0; i < ULIBC_get_num_procs(); ++i) {
    struct cpuinfo_t ci = ULIBC_get_cpuinfo(i);
    fprintf(fp, "ULIBC: CPU[%03d] Processor: %2d, Node: %2d, Core: %2d, SMT: %2d\n",
	    i, ci.id, ci.node, ci.core, ci.smt);
  }
}


static int parse_cpufile(const char *file) {
  int x = -1;
  FILE *fp = fopen(file, "r");
  if (fp) {
    fscanf(fp, "%d", &x);
    fclose(fp);
  }
  return x;
}

static int fill_cpuinfo(struct cpuinfo_t *cpuinfo) {
  char path[PATH_MAX], dirpath[PATH_MAX];
  DIR *dp = NULL, *ldp = NULL;
  struct dirent *dir = NULL, *ldir = NULL;
  int cpuid, coreid, nodeid, num_cpus=0;
  
  /* open cpu files */
  strcpy(dirpath, "/sys/devices/system/cpu");
  dp = opendir(dirpath);
  if (!dp) {
    return 0;
  } else {
    while ( (dir = readdir(dp)) != 0 ) {
      /* scan cpu(core) id */
      cpuid = -1;
      sscanf(dir->d_name, "cpu%d", &cpuid);
      if (cpuid < 0 || MAX_CPUS < cpuid) continue;
    
      /* read core_id */
      sprintf(path, "%s/%s/topology/core_id", dirpath, dir->d_name);
      coreid = parse_cpufile(path);
      if (coreid >= 0) {
	cpuinfo[cpuid].id = cpuid;
	cpuinfo[cpuid].core = coreid;
	++num_cpus;
      }
    }
    closedir(dp);
  }
  
  /* open node files */
  strcpy(dirpath, "/sys/devices/system/node");
  dp = opendir(dirpath);
  if (!dp) {
    return 0;
  } else {
    while ( (dir = readdir(dp)) != 0 ) {
      /* scan node(socket) id */
      nodeid = -1;
      sscanf(dir->d_name, "node%d", &nodeid);
      if (nodeid < 0 || MAX_NODES < nodeid) continue;
        
      /* read cpu(core) id */
      sprintf(path, "%s/%s", dirpath, dir->d_name);
      ldp = opendir(path);
      if ( !ldp ) {
	break;
      } else {
    	while ( (ldir = readdir(ldp)) != NULL ) {
	  if ( !ldir ) continue;
	  if ( !ldir->d_name ) continue;
	  if ( strncmp(ldir->d_name, "cpu", strlen("cpu")) ) continue;
	  if ( !isdigit( *(ldir->d_name+strlen("cpu")) ) ) continue;
	  cpuid = -1;
	  sscanf(ldir->d_name, "cpu%d", &cpuid);
	  if (cpuid < 0 || MAX_CPUS < cpuid) continue;
	  cpuinfo[cpuid].node = nodeid;
	}
	closedir(ldp);
      }
    }
    closedir(dp);
  }
  return num_cpus;
}

static int get_max_nodes(int ncpus, struct cpuinfo_t *cpuinfo) {
  int nodes = 0;
  for (int i = 0; i < ncpus; ++i) {
    nodes = MAX(nodes, cpuinfo[i].node+1);
  }
  return nodes;
}

static int cmpr_int(const void *a, const void *b) {
  int _a = *(int *)a;
  int _b = *(int *)b;
  if ( _a < _b) return -1;
  if ( _a > _b) return  1;
  return 0;
}

static int get_uniq_cores(int ncpus, struct cpuinfo_t *cpuinfo, int cores[MAX_CPUS]) {
  int pos = 0;
  for (int i = 0; i < ncpus; ++i) {
    cores[ pos++ ] = cpuinfo[i].core;
  }
  return uniq(cores, ncpus, sizeof(int), qsort, cmpr_int);
}

static int make_smtid(int ncpus, struct cpuinfo_t *cpuinfo, 
		      int nnodes, int ncores, int uniq_cores[MAX_CPUS]) {
  int map[MAX_CPUS];
  for (int i = 0; i < ncpus; ++i)
    map[i] = -1;
  for (int i = 0; i < ncores; ++i)
    map[ uniq_cores[i] ] = i;
  
  int count[MAX_CPUS];
  for (int i = 0; i < nnodes; ++i) {
    for (int j = 0; j < ncpus; ++j)
      count[j] = 0;
    
    for (int j = 0; j < ncpus; ++j) {
      if (i == cpuinfo[j].node)
	cpuinfo[j].smt = count[ map[ cpuinfo[j].core ] ]++;
    }
  }
  
  /* get max. SMT index */
  int max_smt = 0;
  for (int i = 0; i < ncpus; ++i) {
    max_smt = MAX(max_smt, cpuinfo[i].smt+1);
  }
  return max_smt;
}
