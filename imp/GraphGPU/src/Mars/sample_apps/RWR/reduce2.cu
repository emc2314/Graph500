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

// #define _DEBUG_REDUCE

__device__ void REDUCE_COUNT_FUNC2//(void* key, void* vals, size_t keySize, size_t valCount)
{
  EMIT_COUNT_FUNC(sizeof(int), sizeof(RWR_VAL_T));
}

__device__ void REDUCE_FUNC2//(void* key, void* vals, size_t keySize, size_t valCount, int argi, float argf)
{
  float mixing_c = argf;
  // int number_node = argi;
  float next_rank = 0.0f;
  // float previous_rank = 0.0f;
// #ifdef _DEBUG_REDUCE
//   if(*(int*)key == 0)
//     printf("mixing_c: %f, number_node: %d\n", mixing_c, number_node);
// #endif
  for (int i = 0; i < valCount; i++) {
    RWR_VAL_T* cur_value = (RWR_VAL_T*)GET_VAL_FUNC(vals, i);
    if(cur_value->is_v == false) {
      // previous_rank = cur_value->dst;
    }
    else {
      next_rank += cur_value->dst;
#ifdef _DEBUG_REDUCE
      //printf("reduce: key = %d, next_rank = %f\n", *(int*)key, next_rank);
#endif
    }
  }
  next_rank = next_rank * mixing_c;
#ifdef _DEBUG_REDUCE
  //printf("reduce: key = %d, next_rank = %f\n", *(int*)key, next_rank);
#endif
  RWR_VAL_T* o_val = (RWR_VAL_T*)GET_OUTPUT_BUF(0);
  o_val->is_v = true;
  o_val->dst = next_rank;
  EMIT_FUNC(key, o_val, sizeof(int), sizeof(RWR_VAL_T));
}
#endif //__REDUCE2_CU__
