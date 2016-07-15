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

#ifndef MAX_NUMA_LOCALS
#define MAX_NUMA_LOCALS 256
#endif
#ifndef MAX_NUMA_LOCALS_SQRT
#define MAX_NUMA_LOCALS_SQRT 16
#endif

static struct NUMA_barrier_t {
  int lnp;
  int lrounds;
  
  volatile int sense[MAX_NUMA_LOCALS];
  
  enum tour_rule_t {
    TR_WINNER   = 0,
    TR_LOSER    = 1,
    TR_BYE      = 2,
    TR_CHAMPION = 3,
    TR_DROPOUT  = 4,
  } rule;
  
  volatile struct round_struct {
    enum tour_rule_t rule;
    int *opponent;
    int flag;
  } RS[MAX_NUMA_LOCALS][MAX_NUMA_LOCALS_SQRT];
} *__barrier[MAX_NODES];


static void init_local_tournament_barrier(int node);

int ULIBC_init_numa_barriers(void) {
  if (ULIBC_verbose())
    printf("ULIBC: enable NUMA-barrier using Tournament-barrier\n");
  
  int wakeup_count = 0;
  for (int k = 0; k < ULIBC_get_online_nodes(); ++k) {
    if ( !__barrier[k] ) {
      ++wakeup_count;
      size_t sz = ROUNDUP(sizeof(struct NUMA_barrier_t), ULIBC_align_size());
      __barrier[k] = NUMA_touched_malloc(sz, k);
    }
    init_local_tournament_barrier(k);
  }
  
  if (ULIBC_verbose())
    printf("ULIBC: woke up %d NUMA-barriers\n", wakeup_count);
  return 0;
}


static void init_local_tournament_barrier(int node) {
  /* allocation */
  struct NUMA_barrier_t *nodeNB = (struct NUMA_barrier_t *)__barrier[node];
  nodeNB->lnp = ULIBC_get_online_cores(node);
  nodeNB->lrounds = ceil( log(nodeNB->lnp)/log(2) );
  
  int bool_init = 0;
  for (int j = 0; j < nodeNB->lnp; j++) {
    nodeNB->sense[j] = !bool_init;
    for (int k = 0; k <= nodeNB->lrounds; k++) {
      nodeNB->RS[j][k].flag = 0;
      nodeNB->RS[j][k].rule = -1;
      nodeNB->RS[j][k].opponent = &bool_init;
    }
  }
  
  for(int l = 0; l < nodeNB->lnp; l++) {
    for(int k = 0; k <= nodeNB->lrounds; k++) {
      const long comp_1st = 1UL << k;
      const long comp_2nd = 1UL << (k - !(k<1));
      
      /* set rule */
      if (k > 0) {
	if( (l%comp_1st == 0) &&
	    (comp_1st < nodeNB->lnp) &&
	    (l+comp_2nd < nodeNB->lnp) ) {
	  nodeNB->RS[l][k].rule = TR_WINNER;
	}
	if ( (l%comp_1st == 0) && (l+comp_2nd >= nodeNB->lnp) ) {
	  nodeNB->RS[l][k].rule = TR_BYE;
	}
	if ( l%comp_1st == comp_2nd ) {
	  nodeNB->RS[l][k].rule = TR_LOSER;
	}
	if ( (l == 0) && (comp_1st >= nodeNB->lnp) ) {
	  nodeNB->RS[l][k].rule = TR_CHAMPION;
	}
      } else if (k == 0) {
	nodeNB->RS[l][k].rule = TR_DROPOUT;
      }

      /* set opponent */
      if ( nodeNB->RS[l][k].rule == TR_LOSER ) {
	nodeNB->RS[l][k].opponent = (int *)&nodeNB->RS[l-comp_2nd][k].flag;
      } else if ( nodeNB->RS[l][k].rule == TR_WINNER ||
		  nodeNB->RS[l][k].rule == TR_CHAMPION ) {
	nodeNB->RS[l][k].opponent = (int *)&nodeNB->RS[l+comp_2nd][k].flag;
      }
    }
  }
}


void ULIBC_node_barrier(void) {
  if ( !omp_in_parallel() ) {
    return;
  }

  const struct numainfo_t ni = ULIBC_get_current_numainfo();
  struct NUMA_barrier_t *nodeNB = __barrier[ni.node];
  assert( nodeNB );
  
  volatile int *sense = &(nodeNB->sense[ni.core]);
  int round = 0;
  while (1) {
    if ( nodeNB->RS[ni.core][round].rule == TR_LOSER ) {
      *(nodeNB->RS[ni.core][round]).opponent = *sense;
      while ( nodeNB->RS[ni.core][round].flag != *sense );
      break;
    }
    
    if ( nodeNB->RS[ni.core][round].rule == TR_WINNER ) {
      while( nodeNB->RS[ni.core][round].flag != *sense );
    }

    if ( nodeNB->RS[ni.core][round].rule == TR_CHAMPION ){
      while ( nodeNB->RS[ni.core][round].flag != *sense );
      *( nodeNB->RS[ni.core][round] ).opponent = *sense;
      break;
    }

    if (round < nodeNB->lrounds) round = round + 1;
  }

  //wake up
  while (1) {
    if ( round > 0 ) round = round - 1;
    if ( nodeNB->RS[ni.core][round].rule == TR_WINNER )
      *( nodeNB->RS[ni.core][round] ).opponent = *sense;
    if ( nodeNB->RS[ni.core][round].rule == TR_DROPOUT ) break;
  }

  *sense = !*sense;
}
