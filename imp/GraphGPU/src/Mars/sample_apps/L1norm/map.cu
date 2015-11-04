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

// #define _DEBUG_MAP

__device__ void MAP_COUNT_FUNC//(void *key, void *val, size_t keySize, size_t valSize)
{
  EMIT_INTER_COUNT_FUNC(sizeof(int), sizeof(float));
}

__device__ void MAP_FUNC//(void *key, void val, size_t keySize, size_t valSize)
{
  
  int o_key = 0;
  float raw_val = 0;
  raw_val = fabs(*(float*)val);
#ifdef _DEBUG_MAP
  printf("map: key = %d, value = %f\n", o_key, raw_val);
#endif
  EMIT_INTERMEDIATE_FUNC(&o_key, &raw_val, keySize, valSize);

}
#endif //__MAP_CU__
