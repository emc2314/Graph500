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

#ifndef __MAP_CU__
#define __MAP_CU__

#include "MarsInc.h"
#include "global.h"

__device__ void MAP_COUNT_FUNC//(void *key, void *val, size_t keySize, size_t valSize)
{
  EMIT_INTER_COUNT_FUNC(sizeof(int), sizeof(int));
}

__device__ void MAP_FUNC//(void *key, void val, size_t keySize, size_t valSize)
{
  CC3_VAL_T* iVal = (CC3_VAL_T*)val;
  int o_key = -1;
  int o_val = 1;

  if(iVal->is_v == false) 
    o_key = 0;
  else 
    o_key = 1;

//   if(iVal->dst == 0) 
//     o_key = 0;
//   else 
//     o_key = 1;

  EMIT_INTERMEDIATE_FUNC(&o_key, &o_val, sizeof(int), sizeof(int));
}
#endif //__MAP_CU__
