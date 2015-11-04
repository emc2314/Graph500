/***********************************************************************
 	graphgpu
	Authors: Koichi Shirahata, Hitoshi Sato, Toyotaro Suzumura, and Satoshi Matsuoka

This software is licensed under Apache License, Version 2.0 (the  "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
***********************************************************************/

#ifndef __REDUCE2_CU__
#define __REDUCE2_CU__

#include "MarsInc.h"
#include "global.h"

//#define __NO_REDUCE__
//#define __NO_CONVERGE_CHECK__
// #define _DEBUG_REDUCE


__device__ void REDUCE_COUNT_FUNC2//(void* key, void* vals, size_t keySize, size_t valCount)
{
  EMIT_COUNT_FUNC(sizeof(int), sizeof(PR_VAL_T));
#ifndef __NO_CONVERGE_CHECK__
  EMIT_COUNT_FUNC(sizeof(int), sizeof(PR_VAL_T));
#endif
}

__device__ void REDUCE_FUNC2//(void* key, void* vals, size_t keySize, size_t valCount,
// float argf, int argi)
{
  float mixing_c = argf;  
  int number_node = argi;
  float random_coeff = (1-mixing_c) / (float)number_node;
  float next_rank = 0.0f;
  float previous_rank = 0;
  float converge_threshold = 0.000001;

#ifdef _DEBUG_REDUCE
  if(*(int*)key == 0)
    printf("mixing_c: %f, number_node: %d, random_coeff: %f, converge_threshold: %f\n", 
	mixing_c, number_node, random_coeff, converge_threshold);
#endif

  for (int i = 0; i < valCount; i++) {
    PR_VAL_T* cur_value = (PR_VAL_T*)GET_VAL_FUNC(vals, i);
    if(cur_value->is_v == false) {
      previous_rank = cur_value->dst;
    }
    else {
      next_rank += cur_value->dst;
    }
  }
  next_rank = next_rank * mixing_c + random_coeff;
  PR_VAL_T* o_val = (PR_VAL_T*)GET_OUTPUT_BUF(0);
  o_val->is_v = true;
  o_val->dst = next_rank;
  EMIT_FUNC(key, o_val, sizeof(int), sizeof(PR_VAL_T));

  //CHECK CONVERGENCE
#ifndef __NO_CONVERGE_CHECK__
  float diff = abs(previous_rank - next_rank);
  PR_KEY_T* o_key = (PR_KEY_T*)GET_OUTPUT_BUF(sizeof(PR_VAL_T));
  o_key->src = -1;
  o_val->is_v = false;
  if( diff < converge_threshold ) {
    o_val->dst = 0; // converged
  } else {
    o_val->dst = 1; // chenged
  }
  EMIT_FUNC(o_key, o_val, sizeof(int), sizeof(PR_VAL_T));
#endif
}
#endif //__REDUCE2_CU__
